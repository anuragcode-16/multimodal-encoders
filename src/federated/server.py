"""
Federated learning server implementation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import OrderedDict
from pathlib import Path
import json
from datetime import datetime

from src.models.fedfim import FedFIMModel, create_fedfim_model
from src.federated.client import FederatedClient
from src.federated.aggregation import FedAvgAggregator, FedProxAggregator, FedFIMAggregator
from src.config import FEDERATED_CONFIG, TRAINING_CONFIG, CHECKPOINTS_DIR


class FederatedServer:
    """
    Federated learning server with drift-aware aggregation
    """
    
    def __init__(self, model: FedFIMModel = None, 
                 config=None,
                 aggregation_type: str = 'fedfim',
                 device: str = 'cpu'):
        self.config = config or FEDERATED_CONFIG
        self.device = device
        self.aggregation_type = aggregation_type
        
        # Initialize global model
        if model is None:
            self.global_model = create_fedfim_model(
                num_clients=self.config.num_clients,
                device=device
            )
        else:
            self.global_model = model
        
        # Initialize aggregator
        if aggregation_type == 'fedavg':
            self.aggregator = FedAvgAggregator()
        elif aggregation_type == 'fedprox':
            self.aggregator = FedProxAggregator()
        else:
            self.aggregator = FedFIMAggregator(self.config)
        
        # Tracking
        self.round_number = 0
        self.client_metrics_history = []
        self.global_metrics_history = []
        self.aggregation_weights_history = []
        self.drift_history = []
        self.contribution_history = []
    
    def get_global_parameters(self) -> OrderedDict:
        """Get global model parameters"""
        return self.global_model.get_global_parameters()
    
    def set_global_parameters(self, params: OrderedDict):
        """Set global model parameters"""
        self.global_model.set_global_parameters(params)
    
    def select_clients(self, clients: List[FederatedClient], 
                       fraction: float = None) -> List[FederatedClient]:
        """Select clients for current round"""
        fraction = fraction or self.config.client_selection_ratio
        num_selected = max(
            self.config.min_clients_per_round,
            int(len(clients) * fraction)
        )
        
        # Random selection
        indices = np.random.choice(len(clients), size=num_selected, replace=False)
        return [clients[i] for i in indices]
    
    def distribute_parameters(self, clients: List[FederatedClient]):
        """Distribute global parameters to clients"""
        global_params = self.get_global_parameters()
        for client in clients:
            client.set_parameters(global_params)
    
    def aggregate_updates(self, client_updates: List[Dict]) -> OrderedDict:
        """
        Aggregate client updates
        
        Args:
            client_updates: List of dictionaries with 'params', 'num_samples', 'drift_score', etc.
        
        Returns:
            Aggregated parameters
        """
        aggregated_params = self.aggregator.aggregate(client_updates)
        
        # Track aggregation weights
        if hasattr(self.aggregator, 'last_weights'):
            self.aggregation_weights_history.append(self.aggregator.last_weights.copy())
        
        # Track drift scores
        drifts = [update.get('drift_score', 0) for update in client_updates]
        self.drift_history.append({
            'round': self.round_number,
            'mean_drift': np.mean(drifts),
            'max_drift': np.max(drifts),
            'min_drift': np.min(drifts)
        })
        
        return aggregated_params
    
    def evaluate_global_model(self, test_loader) -> Dict[str, float]:
        """Evaluate global model on test data"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                price = batch['price'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                behavior = batch['behavior'].to(self.device)
                labels = batch['label'].to(self.device)
                client_ids = batch['client_id'].to(self.device)
                
                # Use global model without personalization
                outputs = self.global_model(price, sentiment, behavior, client_ids,
                                            use_personalization=False)
                
                loss = criterion(outputs['direction'], labels)
                total_loss += loss.item()
                
                preds = outputs['direction'].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            'round': self.round_number,
            'loss': total_loss / max(len(test_loader), 1),
            'accuracy': correct / max(total, 1),
            'timestamp': datetime.now().isoformat()
        }
        
        self.global_metrics_history.append(metrics)
        
        return metrics
    
    def train_round(self, clients: List[FederatedClient], 
                    epochs: int = None) -> Dict:
        """
        Execute one federated training round
        
        Args:
            clients: List of available clients
            epochs: Number of local epochs
        
        Returns:
            Dictionary with round metrics
        """
        self.round_number += 1
        epochs = epochs or self.config.local_epochs
        
        # Select participating clients
        selected_clients = self.select_clients(clients)
        
        # Distribute global parameters
        self.distribute_parameters(selected_clients)
        
        # Local training
        client_updates = []
        round_metrics = {
            'round': self.round_number,
            'num_participating': len(selected_clients),
            'client_metrics': []
        }
        
        for client in selected_clients:
            if self.aggregation_type == 'fedprox':
                update = client.local_train_fedprox(
                    self.get_global_parameters(),
                    epochs=epochs,
                    mu=self.config.mu
                )
            else:
                update = client.local_train(epochs=epochs)
            
            # Evaluate client
            eval_metrics = client.evaluate()
            update['val_accuracy'] = eval_metrics['accuracy']
            update['val_loss'] = eval_metrics['loss']
            
            client_updates.append(update)
            round_metrics['client_metrics'].append({
                'client_id': client.client_id,
                'train_loss': update['loss'],
                'train_accuracy': update['accuracy'],
                'val_loss': eval_metrics['loss'],
                'val_accuracy': eval_metrics['accuracy'],
                'drift_score': client.drift_score,
                'contribution_score': client.contribution_score
            })
        
        # Aggregate updates
        aggregated_params = self.aggregate_updates(client_updates)
        self.set_global_parameters(aggregated_params)
        
        # Store metrics
        self.client_metrics_history.append(round_metrics)
        
        return round_metrics
    
    def save_checkpoint(self, path: str = None):
        """Save server state"""
        path = path or str(CHECKPOINTS_DIR / f'server_round_{self.round_number}.pth')
        
        checkpoint = {
            'round_number': self.round_number,
            'global_model_state': self.get_global_parameters(),
            'global_metrics_history': self.global_metrics_history,
            'client_metrics_history': self.client_metrics_history,
            'drift_history': self.drift_history,
            'aggregation_weights_history': self.aggregation_weights_history
        }
        
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path: str):
        """Load server state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.round_number = checkpoint['round_number']
        self.set_global_parameters(checkpoint['global_model_state'])
        self.global_metrics_history = checkpoint.get('global_metrics_history', [])
        self.client_metrics_history = checkpoint.get('client_metrics_history', [])
        self.drift_history = checkpoint.get('drift_history', [])
        self.aggregation_weights_history = checkpoint.get('aggregation_weights_history', [])
    
    def get_training_summary(self) -> Dict:
        """Get summary of federated training"""
        if not self.global_metrics_history:
            return {}
        
        return {
            'total_rounds': self.round_number,
            'final_accuracy': self.global_metrics_history[-1]['accuracy'],
            'final_loss': self.global_metrics_history[-1]['loss'],
            'best_accuracy': max(m['accuracy'] for m in self.global_metrics_history),
            'total_clients': self.config.num_clients,
            'aggregation_type': self.aggregation_type
        }