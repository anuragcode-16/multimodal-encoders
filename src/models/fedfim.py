"""
FedFIM: Complete model architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict

from src.config import MODEL_CONFIG
from src.models.encoders import MultimodalEncoder
from src.models.fusion import AttentionFusion, ConcatFusion, GatedFusion
from src.models.heads import MultiTaskHead, PersonalizationHead


class FedFIMModel(nn.Module):
    """
    FedFIM: Multimodal Personalized Federated Learning Model
    
    Architecture:
    1. Multimodal Encoders (shared globally)
    2. Fusion Module (shared globally)
    3. Prediction Heads (shared globally)
    4. Personalization Adapters (client-specific)
    """
    
    def __init__(self, fusion_type: str = 'attention', use_transformer: bool = False,
                 config=None, num_clients: int = 20):
        super().__init__()
        config = config or MODEL_CONFIG
        self.config = config
        self.num_clients = num_clients
        
        # Shared global components
        self.encoder = MultimodalEncoder(use_transformer=use_transformer, config=config)
        
        # Fusion module
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(config)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(config)
        else:
            self.fusion = ConcatFusion(config)
        
        # Shared prediction heads
        self.heads = MultiTaskHead(config.fusion_output_dim, config)
        
        # Personalization adapters (one per client)
        self.personalization_adapters = nn.ModuleDict({
            str(i): PersonalizationHead(config.fusion_output_dim, config)
            for i in range(num_clients)
        })
        
        # Personalized heads (optional, for more personalization)
        self.personalized_heads = nn.ModuleDict({
            str(i): MultiTaskHead(config.fusion_output_dim, config)
            for i in range(num_clients)
        })
    
    def forward(self, price: torch.Tensor, sentiment: torch.Tensor,
                behavior: torch.Tensor, client_ids: torch.Tensor,
                use_personalization: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional personalization
        
        Args:
            price: (batch, seq_len, price_features)
            sentiment: (batch, sentiment_dim)
            behavior: (batch, behavior_features)
            client_ids: (batch,) client identifiers
            use_personalization: whether to use personalization adapters
        """
        # Encode modalities
        embeddings = self.encoder(price, sentiment, behavior)
        
        # Fuse modalities
        fused = self.fusion(embeddings)
        
        # Apply personalization if requested
        if use_personalization:
            outputs = self._personalized_forward(fused, client_ids)
        else:
            outputs = self.heads(fused)
            outputs['fused'] = fused
        
        # Add embeddings for analysis
        outputs['embeddings'] = embeddings
        
        return outputs
    
    def _personalized_forward(self, fused: torch.Tensor, 
                              client_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply client-specific personalization"""
        batch_size = fused.size(0)
        device = fused.device
        
        # Initialize output tensors
        direction_out = torch.zeros(batch_size, self.config.num_classes_direction, device=device)
        risk_out = torch.zeros(batch_size, device=device)
        action_out = torch.zeros(batch_size, self.config.action_output_dim, device=device)
        
        # Process per-client batches
        unique_clients = client_ids.unique()
        
        for client_id in unique_clients:
            client_id_int = int(client_id.item())
            client_key = str(client_id_int)
            
            # Get mask for this client
            mask = (client_ids == client_id)
            
            # Apply personalization adapter
            adapted = self.personalization_adapters[client_key](fused[mask])
            
            # Get predictions from personalized head
            client_outputs = self.personalized_heads[client_key](adapted)
            
            # Scatter to output tensors
            direction_out[mask] = client_outputs['direction']
            risk_out[mask] = client_outputs['risk']
            action_out[mask] = client_outputs['action']
        
        return {
            'direction': direction_out,
            'risk': risk_out,
            'action': action_out,
            'fused': fused
        }
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters for global aggregation (excludes personalization)"""
        global_params = OrderedDict()
        
        for name, param in self.named_parameters():
            if 'personalization' not in name and 'personalized_heads' not in name:
                global_params[name] = param.data.clone()
        
        return global_params
    
    def set_global_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters from global aggregation"""
        for name, param in self.named_parameters():
            if name in params:
                param.data.copy_(params[name])
    
    def get_client_parameters(self, client_id: int) -> Dict[str, torch.Tensor]:
        """Get personalization parameters for a specific client"""
        client_params = OrderedDict()
        prefix = f'personalization_adapters.{client_id}.'
        head_prefix = f'personalized_heads.{client_id}.'
        
        for name, param in self.named_parameters():
            if name.startswith(prefix) or name.startswith(head_prefix):
                client_params[name] = param.data.clone()
        
        return client_params
    
    def set_client_parameters(self, client_id: int, params: Dict[str, torch.Tensor]):
        """Set personalization parameters for a specific client"""
        for name, param in self.named_parameters():
            if name in params:
                param.data.copy_(params[name])


class FedFIMClient:
    """FedFIM client for local training"""
    
    def __init__(self, client_id: int, model: FedFIMModel, device: str = 'cpu'):
        self.client_id = client_id
        self.model = model
        self.device = device
        
        # Client-specific state
        self.local_data_stats = {}
        self.drift_score = 0.0
        self.contribution_score = 0.0
        self.update_history = []
    
    def local_train(self, train_loader, optimizer, criterion, epochs: int = 5):
        """Perform local training"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in train_loader:
                # Move to device
                price = batch['price'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                behavior = batch['behavior'].to(self.device)
                labels = batch['label'].to(self.device)
                client_ids = batch['client_id'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(price, sentiment, behavior, client_ids,
                                    use_personalization=True)
                
                # Compute loss
                loss = criterion(outputs['direction'], labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'params': self.model.get_global_parameters(),
            'num_samples': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else num_batches
        }
    
    def compute_drift(self, previous_params: Dict[str, torch.Tensor]) -> float:
        """Compute drift score based on parameter changes"""
        current_params = self.model.get_global_parameters()
        
        total_drift = 0.0
        count = 0
        
        for name in current_params:
            if name in previous_params:
                diff = current_params[name] - previous_params[name]
                drift = torch.norm(diff).item()
                total_drift += drift
                count += 1
        
        self.drift_score = total_drift / max(count, 1)
        return self.drift_score
    
    def evaluate(self, val_loader, criterion) -> Dict[str, float]:
        """Evaluate on validation data"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                price = batch['price'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                behavior = batch['behavior'].to(self.device)
                labels = batch['label'].to(self.device)
                client_ids = batch['client_id'].to(self.device)
                
                outputs = self.model(price, sentiment, behavior, client_ids,
                                    use_personalization=True)
                
                loss = criterion(outputs['direction'], labels)
                total_loss += loss.item()
                
                preds = outputs['direction'].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / max(len(val_loader), 1),
            'accuracy': correct / max(total, 1)
        }


def create_fedfim_model(config=None, num_clients: int = 20, device: str = 'cpu') -> FedFIMModel:
    """Factory function to create FedFIM model"""
    model = FedFIMModel(
        fusion_type='attention',
        use_transformer=False,
        config=config,
        num_clients=num_clients
    )
    return model.to(device)