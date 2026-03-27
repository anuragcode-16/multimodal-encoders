"""
Federated learning client implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import OrderedDict

from src.models.fedfim import FedFIMModel
from src.config import FEDERATED_CONFIG, TRAINING_CONFIG


class FederatedClient:
    """
    Federated learning client with drift tracking and contribution scoring
    """
    
    def __init__(self, client_id: int, model: FedFIMModel, 
                 train_loader, val_loader,
                 config=None, device: str = 'cpu'):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or FEDERATED_CONFIG
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG.learning_rate,
            weight_decay=TRAINING_CONFIG.weight_decay
        )
        
        # Tracking variables
        self.previous_params = None
        self.drift_score = 0.0
        self.contribution_score = 0.0
        self.freshness_score = 1.0
        self.consistency_score = 1.0
        self.performance_history = []
        self.update_history = []
        
        # Behavior type for analysis
        self.behavior_type = 'unknown'
    
    def set_parameters(self, params: OrderedDict):
        """Set model parameters from server"""
        self.model.set_global_parameters(params)
        self.previous_params = {k: v.clone() for k, v in params.items()}
    
    def get_parameters(self) -> OrderedDict:
        """Get model parameters for server aggregation"""
        return self.model.get_global_parameters()
    
    def local_train(self, epochs: int = None) -> Dict:
        """
        Perform local training
        
        Returns:
            Dictionary with training metrics and updated parameters
        """
        epochs = epochs or self.config.local_epochs
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch in self.train_loader:
                # Move data to device
                price = batch['price'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                behavior = batch['behavior'].to(self.device)
                labels = batch['label'].to(self.device)
                client_ids = batch['client_id'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(price, sentiment, behavior, client_ids,
                                    use_personalization=True)
                
                # Compute loss
                loss = criterion(outputs['direction'], labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               TRAINING_CONFIG.gradient_clip)
                self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_batches += 1
                
                preds = outputs['direction'].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            total_loss += epoch_loss
            num_batches += epoch_batches
        
        # Compute drift if we have previous parameters
        if self.previous_params is not None:
            self._compute_drift()
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        
        # Track performance
        self.performance_history.append(accuracy)
        
        return {
            'client_id': self.client_id,
            'params': self.get_parameters(),
            'num_samples': len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else total,
            'loss': avg_loss,
            'accuracy': accuracy,
            'drift_score': self.drift_score
        }
    
    def local_train_fedprox(self, global_params: OrderedDict, 
                            epochs: int = None, mu: float = 0.01) -> Dict:
        """
        Local training with FedProx proximal term
        
        Args:
            global_params: Global model parameters from server
            epochs: Number of local epochs
            mu: Proximal term coefficient
        """
        epochs = epochs or self.config.local_epochs
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in self.train_loader:
                price = batch['price'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                behavior = batch['behavior'].to(self.device)
                labels = batch['label'].to(self.device)
                client_ids = batch['client_id'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(price, sentiment, behavior, client_ids,
                                    use_personalization=True)
                
                # Standard loss
                loss = criterion(outputs['direction'], labels)
                
                # FedProx proximal term
                proximal_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in global_params and 'personalization' not in name:
                        proximal_term += torch.norm(param - global_params[name].to(self.device)) ** 2
                
                loss = loss + (mu / 2) * proximal_term
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               TRAINING_CONFIG.gradient_clip)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if self.previous_params is not None:
            self._compute_drift()
        
        return {
            'client_id': self.client_id,
            'params': self.get_parameters(),
            'num_samples': len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else num_batches,
            'loss': total_loss / max(num_batches, 1),
            'drift_score': self.drift_score
        }
    
    def _compute_drift(self):
        """Compute drift score based on parameter changes"""
        current_params = self.get_parameters()
        
        total_drift = 0.0
        count = 0
        
        for name in current_params:
            if name in self.previous_params:
                diff = current_params[name].cpu() - self.previous_params[name].cpu()
                # Use L2 norm normalized by parameter count
                drift = torch.norm(diff).item() / np.sqrt(diff.numel())
                total_drift += drift
                count += 1
        
        self.drift_score = total_drift / max(count, 1)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation data"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
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
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            'loss': total_loss / max(len(self.val_loader), 1),
            'accuracy': correct / max(total, 1),
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def compute_contribution_score(self, global_performance: float,
                                   previous_local_performance: float) -> float:
        """
        Compute contribution score based on multiple factors
        
        Args:
            global_performance: Current global model performance
            previous_local_performance: Previous round local performance
        """
        # Performance improvement
        current_performance = self.performance_history[-1] if self.performance_history else 0
        performance_improvement = max(0, current_performance - previous_local_performance)
        
        # Consistency (low variance in recent performance)
        if len(self.performance_history) > 3:
            recent = self.performance_history[-5:]
            self.consistency_score = 1.0 / (1.0 + np.std(recent))
        
        # Freshness (decay over rounds without updates)
        self.freshness_score = min(1.0, self.freshness_score * 0.95 + 0.05)
        
        # Stability (inverse of drift)
        stability_score = 1.0 / (1.0 + self.drift_score)
        
        # Weighted combination
        alpha = self.config.incentive_alpha
        beta = self.config.incentive_beta
        gamma = self.config.incentive_gamma
        delta = self.config.incentive_delta
        
        self.contribution_score = (
            alpha * performance_improvement +
            beta * self.consistency_score +
            gamma * self.freshness_score +
            delta * stability_score
        )
        
        return self.contribution_score
    
    def get_client_stats(self) -> Dict:
        """Get client statistics for aggregation"""
        return {
            'client_id': self.client_id,
            'num_samples': len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 0,
            'drift_score': self.drift_score,
            'contribution_score': self.contribution_score,
            'freshness_score': self.freshness_score,
            'consistency_score': self.consistency_score,
            'behavior_type': self.behavior_type
        }