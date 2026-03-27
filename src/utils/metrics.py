"""
Comprehensive metrics computation for FedFIM
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error
)


class ClassificationMetrics:
    """Compute classification metrics"""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, 
                y_prob: np.ndarray = None) -> Dict[str, float]:
        """Compute all classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        try:
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        except:
            metrics['precision_macro'] = 0
            metrics['recall_macro'] = 0
            metrics['f1_macro'] = 0
        
        # Weighted metrics
        try:
            metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            metrics['precision_weighted'] = 0
            metrics['recall_weighted'] = 0
            metrics['f1_weighted'] = 0
        
        # AUC (if probabilities provided)
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
            except:
                metrics['auc_macro'] = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics


class FinancialMetrics:
    """Compute financial performance metrics"""
    
    @staticmethod
    def compute_returns(predictions: np.ndarray, actual_returns: np.ndarray,
                        strategy: str = 'long_short') -> np.ndarray:
        """Compute strategy returns"""
        if strategy == 'long_short':
            # Long when predict up, short when predict down
            positions = np.zeros(len(predictions))
            positions[predictions == 0] = 1   # Long
            positions[predictions == 1] = -1  # Short
            positions[predictions == 2] = 0   # Neutral
            returns = positions * actual_returns
        elif strategy == 'long_only':
            positions = (predictions == 0).astype(float)
            returns = positions * actual_returns
        else:
            returns = np.zeros(len(predictions))
        
        return returns
    
    @staticmethod
    def compute(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Compute financial metrics from returns"""
        metrics = {}
        
        # Basic returns
        metrics['total_return'] = np.sum(returns)
        metrics['mean_return'] = np.mean(returns)
        
        # Volatility
        metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Sharpe Ratio
        excess_return = metrics['mean_return'] * 252 - risk_free_rate
        metrics['sharpe_ratio'] = excess_return / (metrics['volatility'] + 1e-8)
        
        # Cumulative returns
        cumulative = np.cumprod(1 + returns)
        metrics['cumulative_return'] = cumulative[-1] - 1 if len(cumulative) > 0 else 0
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        metrics['max_drawdown'] = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        metrics['win_rate'] = np.mean(returns > 0)
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = gains / (losses + 1e-8)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        metrics['sortino_ratio'] = excess_return / downside_std
        
        return metrics


class FederatedMetrics:
    """Compute federated learning specific metrics"""
    
    @staticmethod
    def communication_cost(model_size: int, num_clients: int, 
                           num_rounds: int) -> Dict[str, float]:
        """Compute communication cost"""
        # Assuming bidirectional communication
        total_bytes = 2 * model_size * num_clients * num_rounds
        
        return {
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'per_round_mb': total_bytes / (1024 * 1024 * num_rounds),
            'per_client_mb': total_bytes / (1024 * 1024 * num_clients)
        }
    
    @staticmethod
    def convergence_speed(loss_history: List[float], 
                          target_loss: float = 0.1) -> Dict[str, float]:
        """Analyze convergence speed"""
        if not loss_history:
            return {'convergence_rounds': -1, 'final_loss': 0}
        
        # Find convergence round
        convergence_round = -1
        for i, loss in enumerate(loss_history):
            if loss <= target_loss:
                convergence_round = i + 1
                break
        
        # Compute convergence rate
        losses = np.array(loss_history)
        if len(losses) > 1:
            rates = np.diff(losses) / losses[:-1]
            avg_rate = np.mean(rates)
        else:
            avg_rate = 0
        
        return {
            'convergence_rounds': convergence_round if convergence_round > 0 else len(loss_history),
            'final_loss': losses[-1],
            'convergence_rate': avg_rate,
            'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else 0
        }
    
    @staticmethod
    def personalization_gain(global_acc: float, 
                             personalized_accs: Dict[int, float]) -> Dict[str, float]:
        """Compute personalization improvement"""
        if not personalized_accs:
            return {'mean_gain': 0, 'max_gain': 0}
        
        gains = {cid: acc - global_acc for cid, acc in personalized_accs.items()}
        
        return {
            'mean_gain': np.mean(list(gains.values())),
            'max_gain': max(gains.values()),
            'min_gain': min(gains.values()),
            'std_gain': np.std(list(gains.values())),
            'improved_clients': sum(1 for g in gains.values() if g > 0),
            'gain_distribution': gains
        }


class DriftMetrics:
    """Compute drift-related metrics"""
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Compute KL divergence"""
        p = p + eps
        q = q + eps
        return np.sum(p * np.log(p / q))
    
    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Compute Jensen-Shannon divergence"""
        p = p + eps
        q = q + eps
        m = 0.5 * (p + q)
        return 0.5 * (DriftMetrics.kl_divergence(p, m) + DriftMetrics.kl_divergence(q, m))
    
    @staticmethod
    def compute_drift_resilience(accuracy_before: float, 
                                  accuracy_during: float,
                                  accuracy_after: float) -> Dict[str, float]:
        """Compute drift resilience metrics"""
        return {
            'accuracy_drop': accuracy_before - accuracy_during,
            'recovery_rate': (accuracy_after - accuracy_during) / (accuracy_before - accuracy_during + 1e-8),
            'resilience_score': accuracy_during / accuracy_before if accuracy_before > 0 else 0
        }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_prob: np.ndarray = None,
                        returns: np.ndarray = None) -> Dict[str, float]:
    """Compute all metrics at once"""
    metrics = {}
    
    # Classification metrics
    metrics.update(ClassificationMetrics.compute(y_true, y_pred, y_prob))
    
    # Financial metrics
    if returns is not None:
        metrics.update(FinancialMetrics.compute(returns))
    
    return metrics