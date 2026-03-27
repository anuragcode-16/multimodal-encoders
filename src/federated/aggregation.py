"""
Federated aggregation strategies
"""
import torch
import numpy as np
from typing import Dict, List, Optional, OrderedDict
from collections import OrderedDict as OD

from src.config import FEDERATED_CONFIG


class BaseAggregator:
    """Base class for aggregation strategies"""
    
    def __init__(self):
        self.last_weights = {}
    
    def aggregate(self, client_updates: List[Dict]) -> OrderedDict:
        """Aggregate client updates"""
        raise NotImplementedError


class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging"""
    
    def aggregate(self, client_updates: List[Dict]) -> OrderedDict:
        """
        Weighted average based on number of samples
        
        Args:
            client_updates: List of dicts with 'params' and 'num_samples'
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated parameters
        aggregated = OD()
        
        # Get parameter names from first update
        first_params = client_updates[0]['params']
        
        for name in first_params:
            weighted_sum = None
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                
                if weighted_sum is None:
                    weighted_sum = update['params'][name].clone() * weight
                else:
                    weighted_sum += update['params'][name].clone() * weight
            
            aggregated[name] = weighted_sum
        
        # Store weights for tracking
        self.last_weights = {
            update['client_id']: update['num_samples'] / total_samples
            for update in client_updates
        }
        
        return aggregated


class FedProxAggregator(BaseAggregator):
    """FedProx aggregation (same as FedAvg but clients use proximal term)"""
    
    def aggregate(self, client_updates: List[Dict]) -> OrderedDict:
        """Same as FedAvg aggregation"""
        aggregator = FedAvgAggregator()
        result = aggregator.aggregate(client_updates)
        self.last_weights = aggregator.last_weights
        return result


class FedFIMAggregator(BaseAggregator):
    """
    FedFIM: Drift-aware, incentive-compatible aggregation
    
    Weights clients based on:
    1. Data quantity
    2. Drift score (lower drift = higher weight)
    3. Contribution score
    4. Freshness
    5. Stability
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FEDERATED_CONFIG
        
        # Aggregation weight factors
        self.data_weight = 0.3
        self.drift_weight = 0.25
        self.contribution_weight = 0.25
        self.stability_weight = 0.2
    
    def aggregate(self, client_updates: List[Dict]) -> OrderedDict:
        """
        Adaptive aggregation based on client quality metrics
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Compute quality scores for each client
        quality_scores = self._compute_quality_scores(client_updates)
        
        # Normalize to get aggregation weights
        total_quality = sum(quality_scores.values())
        aggregation_weights = {
            cid: score / total_quality 
            for cid, score in quality_scores.items()
        }
        
        # Store weights
        self.last_weights = aggregation_weights.copy()
        
        # Aggregate parameters
        aggregated = OD()
        first_params = client_updates[0]['params']
        
        for name in first_params:
            weighted_sum = None
            
            for update in client_updates:
                client_id = update['client_id']
                weight = aggregation_weights[client_id]
                
                if weighted_sum is None:
                    weighted_sum = update['params'][name].clone() * weight
                else:
                    weighted_sum += update['params'][name].clone() * weight
            
            aggregated[name] = weighted_sum
        
        return aggregated
    
    def _compute_quality_scores(self, client_updates: List[Dict]) -> Dict[int, float]:
        """Compute quality score for each client"""
        scores = {}
        
        # Normalize data sizes
        max_samples = max(u['num_samples'] for u in client_updates)
        
        # Normalize drift scores (inverse)
        drifts = [u.get('drift_score', 0) for u in client_updates]
        max_drift = max(drifts) if max(drifts) > 0 else 1.0
        
        for update in client_updates:
            client_id = update['client_id']
            
            # Data quantity score
            data_score = update['num_samples'] / max_samples
            
            # Drift score (lower drift = higher score)
            drift = update.get('drift_score', 0)
            drift_score = 1.0 - min(drift / max_drift, 1.0)
            
            # Contribution score
            contrib_score = update.get('contribution_score', 0.5)
            
            # Stability score (based on validation performance)
            val_acc = update.get('val_accuracy', 0.5)
            stability_score = min(1.0, val_acc * 2)  # Scale to [0, 1]
            
            # Combined quality score
            quality = (
                self.data_weight * data_score +
                self.drift_weight * drift_score +
                self.contribution_weight * contrib_score +
                self.stability_weight * stability_score
            )
            
            scores[client_id] = max(quality, 0.01)  # Minimum weight
        
        return scores


class AdaptiveAggregator(FedFIMAggregator):
    """
    Adaptive aggregator that adjusts weights based on historical performance
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.client_history = {}  # Track historical performance
        self.adaptation_rate = 0.1
    
    def aggregate(self, client_updates: List[Dict]) -> OrderedDict:
        """Aggregate with adaptive weighting"""
        # Update history
        for update in client_updates:
            client_id = update['client_id']
            if client_id not in self.client_history:
                self.client_history[client_id] = {
                    'accuracies': [],
                    'drifts': [],
                    'contributions': []
                }
            
            self.client_history[client_id]['accuracies'].append(
                update.get('val_accuracy', 0.5)
            )
            self.client_history[client_id]['drifts'].append(
                update.get('drift_score', 0)
            )
            self.client_history[client_id]['contributions'].append(
                update.get('contribution_score', 0.5)
            )
        
        # Compute base scores
        quality_scores = self._compute_quality_scores(client_updates)
        
        # Adjust based on historical trends
        for update in client_updates:
            client_id = update['client_id']
            history = self.client_history.get(client_id, {})
            
            # Check if performance is improving
            if len(history.get('accuracies', [])) > 2:
                recent_acc = history['accuracies'][-3:]
                trend = recent_acc[-1] - recent_acc[0]
                
                # Boost weight for improving clients
                if trend > 0:
                    quality_scores[client_id] *= (1 + self.adaptation_rate * trend)
        
        # Normalize
        total_quality = sum(quality_scores.values())
        aggregation_weights = {
            cid: score / total_quality 
            for cid, score in quality_scores.items()
        }
        
        self.last_weights = aggregation_weights.copy()
        
        # Aggregate parameters
        aggregated = OD()
        first_params = client_updates[0]['params']
        
        for name in first_params:
            weighted_sum = None
            
            for update in client_updates:
                client_id = update['client_id']
                weight = aggregation_weights[client_id]
                
                if weighted_sum is None:
                    weighted_sum = update['params'][name].clone() * weight
                else:
                    weighted_sum += update['params'][name].clone() * weight
            
            aggregated[name] = weighted_sum
        
        return aggregated