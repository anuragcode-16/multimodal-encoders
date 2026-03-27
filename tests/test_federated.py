"""
Unit tests for federated learning components
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.federated.aggregation import FedAvgAggregator, FedFIMAggregator
from src.federated.drift import DriftDetector, DriftAdapter
from src.federated.incentive import IncentiveMechanism
from src.config import FEDERATED_CONFIG


class TestAggregation:
    """Test aggregation strategies"""
    
    def test_fedavg_aggregation(self):
        """Test FedAvg aggregation"""
        aggregator = FedAvgAggregator()
        
        # Create mock client updates
        client_updates = []
        for i in range(5):
            params = OrderedDict([
                ('layer1.weight', torch.randn(10, 10)),
                ('layer1.bias', torch.randn(10))
            ])
            client_updates.append({
                'client_id': i,
                'params': params,
                'num_samples': 100,
                'drift_score': 0.1
            })
        
        aggregated = aggregator.aggregate(client_updates)
        
        assert 'layer1.weight' in aggregated
        assert 'layer1.bias' in aggregated
        assert aggregated['layer1.weight'].shape == (10, 10)
    
    def test_fedfim_aggregation(self):
        """Test FedFIM drift-aware aggregation"""
        aggregator = FedFIMAggregator()
        
        # Create mock updates with varying quality
        client_updates = []
        for i in range(5):
            params = OrderedDict([
                ('layer1.weight', torch.randn(10, 10)),
                ('layer1.bias', torch.randn(10))
            ])
            client_updates.append({
                'client_id': i,
                'params': params,
                'num_samples': 100 + i * 50,
                'drift_score': 0.1 + i * 0.1,
                'val_accuracy': 0.7 - i * 0.05,
                'contribution_score': 0.5
            })
        
        aggregated = aggregator.aggregate(client_updates)
        
        assert 'layer1.weight' in aggregated
        assert aggregator.last_weights is not None


class TestDriftDetection:
    """Test drift detection mechanisms"""
    
    def test_parameter_drift(self):
        """Test parameter drift computation"""
        detector = DriftDetector()
        
        current = {'w1': torch.tensor([1.0, 2.0, 3.0])}
        previous = {'w1': torch.tensor([1.1, 2.1, 3.1])}
        
        drift = detector.compute_parameter_drift(0, current, previous)
        
        assert isinstance(drift, float)
        assert drift >= 0
    
    def test_distribution_drift(self):
        """Test distribution drift computation"""
        detector = DriftDetector()
        
        np.random.seed(42)
        current = np.random.randn(100, 5)
        reference = np.random.randn(100, 5) + 0.5
        
        metrics = detector.compute_distribution_drift(current, reference)
        
        assert 'kl_divergence' in metrics
        assert 'js_divergence' in metrics
        assert 'wasserstein_distance' in metrics
    
    def test_concept_drift_detection(self):
        """Test concept drift detection"""
        detector = DriftDetector(threshold=0.2)
        
        # Simulate performance drop
        performance = [0.8] * 10 + [0.6] * 5
        
        result = detector.detect_concept_drift(performance)
        
        assert 'drift_detected' in result
        assert 'drift_score' in result


class TestIncentiveMechanism:
    """Test incentive mechanism"""
    
    def test_contribution_scoring(self):
        """Test contribution score computation"""
        incentive = IncentiveMechanism()
        
        metrics = {
            'val_accuracy': 0.8,
            'train_accuracy': 0.85,
            'freshness_score': 1.0,
            'drift_score': 0.1
        }
        
        score = incentive.compute_contribution_score(0, metrics)
        
        assert 0 <= score <= 1
    
    def test_reward_computation(self):
        """Test reward computation"""
        incentive = IncentiveMechanism()
        
        client_metrics = {
            0: {'val_accuracy': 0.8, 'train_accuracy': 0.85, 'drift_score': 0.1},
            1: {'val_accuracy': 0.7, 'train_accuracy': 0.75, 'drift_score': 0.2},
            2: {'val_accuracy': 0.75, 'train_accuracy': 0.8, 'drift_score': 0.15}
        }
        
        rewards = incentive.compute_all_rewards(client_metrics)
        
        assert len(rewards) == 3
        assert all(r >= 0 for r in rewards.values())
    
    def test_free_rider_detection(self):
        """Test free rider detection"""
        incentive = IncentiveMechanism()
        
        # Add low contributions
        for i in range(5):
            incentive.contribution_history[i] = [0.05, 0.08, 0.04, 0.06, 0.07]
        
        free_riders = incentive.detect_free_riders(threshold=0.1)
        
        assert len(free_riders) == 5
    
    def test_fairness_score(self):
        """Test fairness score computation"""
        incentive = IncentiveMechanism()
        
        # Set some rewards
        incentive.total_rewards = {i: 0.2 for i in range(5)}
        
        fairness = incentive.compute_fairness_score()
        
        assert 0 <= fairness <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])