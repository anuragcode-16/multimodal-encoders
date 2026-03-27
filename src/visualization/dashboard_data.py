"""
Dashboard data provider module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CHECKPOINTS_DIR, FEDERATED_CONFIG
from src.utils.helpers import load_json


class DashboardDataProvider:
    """
    Provides data for the dashboard from training results or demo data
    """
    
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir or CHECKPOINTS_DIR / 'results')
        self.training_history = None
        self.client_metrics = None
        self.drift_history = None
        self.incentive_rewards = None
    
    def load_results(self):
        """Load training results from files"""
        history_path = self.results_dir / 'training_history.json'
        
        if history_path.exists():
            data = load_json(str(history_path))
            self.training_history = data.get('global_metrics', [])
            self.client_metrics = data.get('client_metrics', [])
            self.drift_history = data.get('drift_scores', [])
            self.incentive_rewards = data.get('incentive_rewards', [])
            return True
        
        return False
    
    def get_summary_metrics(self) -> Dict:
        """Get summary metrics for dashboard"""
        if not self.training_history:
            return self._get_demo_summary()
        
        last_metrics = self.training_history[-1] if self.training_history else {}
        
        return {
            'final_accuracy': last_metrics.get('accuracy', 0),
            'final_loss': last_metrics.get('loss', 0),
            'total_rounds': len(self.training_history),
            'best_accuracy': max(m.get('accuracy', 0) for m in self.training_history) if self.training_history else 0,
            'num_clients': FEDERATED_CONFIG.num_clients
        }
    
    def get_training_curves(self) -> Dict[str, List]:
        """Get training curves data"""
        if not self.training_history:
            return self._get_demo_training_curves()
        
        return {
            'rounds': list(range(1, len(self.training_history) + 1)),
            'accuracy': [m.get('accuracy', 0) for m in self.training_history],
            'loss': [m.get('loss', 0) for m in self.training_history]
        }
    
    def get_client_contributions(self) -> Dict[int, float]:
        """Get client contribution scores"""
        if not self.incentive_rewards:
            return self._get_demo_contributions()
        
        # Get latest rewards
        if self.incentive_rewards:
            last_rewards = self.incentive_rewards[-1]
            return {int(k): v for k, v in last_rewards.items()}
        
        return {}
    
    def get_drift_data(self) -> Dict[str, List]:
        """Get drift scores history"""
        if not self.drift_history:
            return self._get_demo_drift()
        
        rounds = list(range(1, len(self.drift_history) + 1))
        mean_drifts = []
        max_drifts = []
        
        for drift_dict in self.drift_history:
            if drift_dict:
                values = list(drift_dict.values())
                mean_drifts.append(np.mean(values))
                max_drifts.append(max(values))
            else:
                mean_drifts.append(0)
                max_drifts.append(0)
        
        return {
            'rounds': rounds,
            'mean_drift': mean_drifts,
            'max_drift': max_drifts
        }
    
    def get_personalization_data(self) -> Dict:
        """Get personalization performance data"""
        if not self.client_metrics:
            return self._get_demo_personalization()
        
        global_acc = self.training_history[-1].get('accuracy', 0) if self.training_history else 0
        
        personalized = {}
        for round_metrics in self.client_metrics:
            for client_metric in round_metrics:
                client_id = client_metric.get('client_id')
                val_acc = client_metric.get('val_accuracy', 0)
                if client_id is not None:
                    personalized[client_id] = max(personalized.get(client_id, 0), val_acc)
        
        return {
            'global_accuracy': global_acc,
            'personalized_accuracies': personalized,
            'gains': {k: v - global_acc for k, v in personalized.items()}
        }
    
    # Demo data generators
    
    def _get_demo_summary(self) -> Dict:
        """Generate demo summary"""
        return {
            'final_accuracy': 0.82,
            'final_loss': 0.45,
            'total_rounds': 50,
            'best_accuracy': 0.85,
            'num_clients': FEDERATED_CONFIG.num_clients
        }
    
    def _get_demo_training_curves(self) -> Dict[str, List]:
        """Generate demo training curves"""
        rounds = list(range(1, 51))
        return {
            'rounds': rounds,
            'accuracy': [0.5 + 0.006*r + np.random.uniform(-0.02, 0.02) for r in rounds],
            'loss': [1.0 - 0.01*r + np.random.uniform(-0.05, 0.05) for r in rounds]
        }
    
    def _get_demo_contributions(self) -> Dict[int, float]:
        """Generate demo contributions"""
        np.random.seed(42)
        return {i: np.random.uniform(0.3, 1.0) for i in range(FEDERATED_CONFIG.num_clients)}
    
    def _get_demo_drift(self) -> Dict[str, List]:
        """Generate demo drift data"""
        rounds = list(range(1, 51))
        return {
            'rounds': rounds,
            'mean_drift': [0.1 + 0.15*np.sin(r/8) + np.random.uniform(-0.02, 0.02) for r in rounds],
            'max_drift': [0.2 + 0.2*np.sin(r/8) + np.random.uniform(-0.03, 0.03) for r in rounds]
        }
    
    def _get_demo_personalization(self) -> Dict:
        """Generate demo personalization data"""
        np.random.seed(42)
        global_acc = 0.78
        personalized = {i: global_acc + np.random.uniform(-0.05, 0.12) 
                       for i in range(FEDERATED_CONFIG.num_clients)}
        
        return {
            'global_accuracy': global_acc,
            'personalized_accuracies': personalized,
            'gains': {k: v - global_acc for k, v in personalized.items()}
        }


# Singleton instance
_data_provider = None


def get_data_provider() -> DashboardDataProvider:
    """Get data provider instance"""
    global _data_provider
    if _data_provider is None:
        _data_provider = DashboardDataProvider()
        _data_provider.load_results()
    return _data_provider