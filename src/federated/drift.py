"""
Drift detection and adaptation for federated learning
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import deque

from src.config import FEDERATED_CONFIG


class DriftDetector:
    """
    Detects distribution drift in client updates and data
    """
    
    def __init__(self, window_size: int = 10, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
        
        # History tracking
        self.param_history = {}  # client_id -> deque of parameters
        self.drift_scores = {}   # client_id -> drift scores
        self.global_drift_history = []
    
    def compute_parameter_drift(self, client_id: int, 
                                 current_params: Dict[str, torch.Tensor],
                                 previous_params: Dict[str, torch.Tensor]) -> float:
        """
        Compute drift score based on parameter changes
        """
        total_drift = 0.0
        count = 0
        
        for name in current_params:
            if name in previous_params:
                current = current_params[name].flatten()
                previous = previous_params[name].flatten()
                
                # L2 distance normalized
                l2_dist = torch.norm(current - previous).item()
                norm_factor = torch.norm(previous).item() + 1e-8
                
                drift = l2_dist / norm_factor
                total_drift += drift
                count += 1
        
        avg_drift = total_drift / max(count, 1)
        
        # Update history
        if client_id not in self.drift_scores:
            self.drift_scores[client_id] = deque(maxlen=self.window_size)
        self.drift_scores[client_id].append(avg_drift)
        
        return avg_drift
    
    def compute_distribution_drift(self, current_data: np.ndarray,
                                    reference_data: np.ndarray) -> Dict[str, float]:
        """
        Compute distribution drift using statistical tests
        """
        drift_metrics = {}
        
        # KL Divergence (for discrete distributions)
        # Bin data for KL computation
        n_bins = 20
        current_hist, _ = np.histogramdd(current_data, bins=n_bins)
        reference_hist, _ = np.histogramdd(reference_data, bins=n_bins)
        
        # Normalize to probabilities
        current_prob = current_hist.flatten() / (current_hist.sum() + 1e-10)
        reference_prob = reference_hist.flatten() / (reference_hist.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        current_prob = current_prob + eps
        reference_prob = reference_prob + eps
        
        # KL Divergence
        kl_div = np.sum(reference_prob * np.log(reference_prob / current_prob))
        drift_metrics['kl_divergence'] = kl_div
        
        # Jensen-Shannon Divergence
        m = 0.5 * (current_prob + reference_prob)
        js_div = 0.5 * (np.sum(reference_prob * np.log(reference_prob / m)) +
                        np.sum(current_prob * np.log(current_prob / m)))
        drift_metrics['js_divergence'] = js_div
        
        # Wasserstein distance (for continuous)
        if current_data.ndim == 1:
            wasserstein = stats.wasserstein_distance(current_data, reference_data)
        else:
            # Use mean across features
            wasserstein = np.mean([
                stats.wasserstein_distance(current_data[:, i], reference_data[:, i])
                for i in range(min(current_data.shape[1], 5))
            ])
        drift_metrics['wasserstein_distance'] = wasserstein
        
        # Kolmogorov-Smirnov test
        if current_data.ndim == 1:
            ks_stat, ks_pvalue = stats.ks_2samp(current_data, reference_data)
        else:
            # Use first feature
            ks_stat, ks_pvalue = stats.ks_2samp(current_data[:, 0], reference_data[:, 0])
        drift_metrics['ks_statistic'] = ks_stat
        drift_metrics['ks_pvalue'] = ks_pvalue
        
        return drift_metrics
    
    def detect_concept_drift(self, performance_history: List[float],
                              window: int = None) -> Dict[str, float]:
        """
        Detect concept drift from performance degradation
        """
        window = window or self.window_size
        
        if len(performance_history) < window:
            return {'drift_detected': False, 'drift_score': 0.0}
        
        recent = performance_history[-window:]
        earlier = performance_history[-2*window:-window] if len(performance_history) >= 2*window else recent
        
        # Compute statistics
        recent_mean = np.mean(recent)
        recent_std = np.std(recent)
        earlier_mean = np.mean(earlier)
        
        # Performance drop
        performance_drop = earlier_mean - recent_mean
        
        # Drift score
        drift_score = max(0, performance_drop) / (recent_std + 1e-8)
        
        # Detect significant drift
        drift_detected = drift_score > self.threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'performance_drop': performance_drop,
            'recent_mean': recent_mean,
            'earlier_mean': earlier_mean
        }
    
    def compute_global_drift(self, client_drifts: Dict[int, float]) -> float:
        """
        Compute global drift score across all clients
        """
        if not client_drifts:
            return 0.0
        
        drifts = list(client_drifts.values())
        
        # Mean drift
        mean_drift = np.mean(drifts)
        
        # Fraction of high-drift clients
        high_drift_fraction = sum(1 for d in drifts if d > self.threshold) / len(drifts)
        
        # Combined global drift
        global_drift = 0.7 * mean_drift + 0.3 * high_drift_fraction
        
        self.global_drift_history.append({
            'mean_drift': mean_drift,
            'high_drift_fraction': high_drift_fraction,
            'global_drift': global_drift
        })
        
        return global_drift
    
    def get_drift_adaptation_weights(self, client_drifts: Dict[int, float]) -> Dict[int, float]:
        """
        Compute adaptation weights based on drift scores
        Lower drift -> higher weight
        """
        if not client_drifts:
            return {}
        
        # Invert and normalize drift scores
        max_drift = max(client_drifts.values()) + 1e-8
        
        weights = {
            cid: 1.0 - (drift / max_drift)
            for cid, drift in client_drifts.items()
        }
        
        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {cid: w / total for cid, w in weights.items()}
        
        return weights


class DriftAdapter:
    """
    Adapts model and aggregation based on detected drift
    """
    
    def __init__(self, config=None):
        self.config = config or FEDERATED_CONFIG
        self.drift_detector = DriftDetector(
            threshold=config.drift_threshold if config else 0.3
        )
        self.adaptation_history = []
    
    def adapt_learning_rate(self, base_lr: float, drift_score: float) -> float:
        """
        Adjust learning rate based on drift
        Higher drift -> higher learning rate (faster adaptation)
        """
        adaptation_factor = 1 + self.config.drift_adaptation_rate * drift_score
        adapted_lr = base_lr * adaptation_factor
        return min(adapted_lr, base_lr * 3)  # Cap at 3x base lr
    
    def adapt_aggregation_weights(self, base_weights: Dict[int, float],
                                   drift_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Adjust aggregation weights based on drift
        """
        adapted = {}
        
        for client_id, weight in base_weights.items():
            drift = drift_scores.get(client_id, 0)
            
            # Reduce weight for high-drift clients
            if drift > self.drift_detector.threshold:
                adaptation = 1.0 - (drift - self.drift_detector.threshold)
                adaptation = max(0.1, adaptation)  # Minimum 10% weight
            else:
                adaptation = 1.0
            
            adapted[client_id] = weight * adaptation
        
        # Renormalize
        total = sum(adapted.values())
        adapted = {cid: w / total for cid, w in adapted.items()}
        
        return adapted
    
    def detect_regime_change(self, market_data: np.ndarray,
                              window: int = 20) -> Dict[str, float]:
        """
        Detect market regime changes
        """
        if len(market_data) < 2 * window:
            return {'regime_change': False, 'score': 0.0}
        
        # Compute volatility in windows
        recent_returns = market_data[-window:]
        earlier_returns = market_data[-2*window:-window]
        
        recent_vol = np.std(recent_returns)
        earlier_vol = np.std(earlier_returns)
        
        # Volatility regime change
        vol_change = abs(recent_vol - earlier_vol) / (earlier_vol + 1e-8)
        
        # Mean return change
        recent_mean = np.mean(recent_returns)
        earlier_mean = np.mean(earlier_returns)
        mean_change = abs(recent_mean - earlier_mean)
        
        # Combined regime change score
        regime_score = 0.6 * vol_change + 0.4 * min(mean_change * 10, 1.0)
        
        return {
            'regime_change': regime_score > 0.5,
            'score': regime_score,
            'volatility_change': vol_change,
            'mean_change': mean_change
        }
    
    def log_adaptation(self, round_num: int, adaptation_info: Dict):
        """Log adaptation for analysis"""
        self.adaptation_history.append({
            'round': round_num,
            **adaptation_info
        })