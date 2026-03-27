"""
Incentive mechanism for federated learning
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.config import FEDERATED_CONFIG


class IncentiveMechanism:
    """
    Incentive mechanism for rewarding client contributions
    
    Computes rewards based on:
    1. Performance contribution
    2. Data quality
    3. Consistency
    4. Freshness
    5. Stability under drift
    """
    
    def __init__(self, config=None):
        self.config = config or FEDERATED_CONFIG
        
        # Contribution history
        self.contribution_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        
        # Tracking
        self.total_rewards = defaultdict(float)
        self.client_rankings = {}
    
    def compute_contribution_score(self, client_id: int,
                                    metrics: Dict[str, float],
                                    historical_performance: List[float] = None) -> float:
        """
        Compute contribution score for a client
        
        Args:
            client_id: Client identifier
            metrics: Dictionary with performance metrics
            historical_performance: List of past performances
        
        Returns:
            Contribution score in [0, 1]
        """
        alpha = self.config.incentive_alpha  # Performance weight
        beta = self.config.incentive_beta    # Consistency weight
        gamma = self.config.incentive_gamma  # Freshness weight
        delta = self.config.incentive_delta  # Stability weight
        
        # Performance improvement
        val_accuracy = metrics.get('val_accuracy', 0.5)
        train_accuracy = metrics.get('train_accuracy', 0.5)
        performance_score = (val_accuracy + train_accuracy) / 2
        
        # Consistency (low variance in performance)
        if historical_performance and len(historical_performance) > 3:
            consistency = 1.0 / (1.0 + np.std(historical_performance[-10:]))
        else:
            consistency = 0.5
        
        # Freshness (recent updates valued more)
        freshness = metrics.get('freshness_score', 1.0)
        
        # Stability under drift
        drift_score = metrics.get('drift_score', 0)
        stability = 1.0 / (1.0 + drift_score)
        
        # Weighted combination
        contribution = (
            alpha * performance_score +
            beta * consistency +
            gamma * freshness +
            delta * stability
        )
        
        # Store in history
        self.contribution_history[client_id].append(contribution)
        
        return contribution
    
    def compute_reward(self, client_id: int, 
                       contribution_score: float,
                       total_contribution: float) -> float:
        """
        Compute reward for a client
        
        Uses Shapley-value-inspired distribution
        """
        if total_contribution <= 0:
            return 0.0
        
        # Proportional reward
        base_reward = contribution_score / total_contribution
        
        # Apply bonus for high contributors
        history = self.contribution_history[client_id]
        if len(history) > 5:
            avg_contribution = np.mean(history[-10:])
            if avg_contribution > 0.6:
                # Bonus for consistently high contributors
                base_reward *= (1 + 0.1 * (avg_contribution - 0.6))
        
        reward = base_reward
        self.reward_history[client_id].append(reward)
        self.total_rewards[client_id] += reward
        
        return reward
    
    def compute_all_rewards(self, client_metrics: Dict[int, Dict]) -> Dict[int, float]:
        """
        Compute rewards for all clients
        
        Args:
            client_metrics: Dict mapping client_id to metrics dict
        
        Returns:
            Dict mapping client_id to reward
        """
        # Compute contribution scores
        contributions = {}
        for client_id, metrics in client_metrics.items():
            history = self.contribution_history.get(client_id, [])
            score = self.compute_contribution_score(client_id, metrics, history)
            contributions[client_id] = score
        
        # Total contribution
        total = sum(contributions.values())
        
        # Compute rewards
        rewards = {}
        for client_id, contrib in contributions.items():
            rewards[client_id] = self.compute_reward(client_id, contrib, total)
        
        # Update rankings
        sorted_clients = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        self.client_rankings = {cid: rank + 1 for rank, (cid, _) in enumerate(sorted_clients)}
        
        return rewards
    
    def get_client_report(self, client_id: int) -> Dict:
        """
        Get detailed report for a client
        """
        return {
            'client_id': client_id,
            'total_reward': self.total_rewards[client_id],
            'average_contribution': np.mean(self.contribution_history[client_id]) if self.contribution_history[client_id] else 0,
            'recent_contributions': self.contribution_history[client_id][-10:],
            'recent_rewards': self.reward_history[client_id][-10:],
            'ranking': self.client_rankings.get(client_id, 0)
        }
    
    def get_summary(self) -> Dict:
        """
        Get summary of incentive mechanism
        """
        all_contributions = [c for hist in self.contribution_history.values() for c in hist]
        all_rewards = [r for hist in self.reward_history.values() for r in hist]
        
        return {
            'total_rounds': len(all_contributions) // max(len(self.contribution_history), 1),
            'total_clients': len(self.contribution_history),
            'mean_contribution': np.mean(all_contributions) if all_contributions else 0,
            'std_contribution': np.std(all_contributions) if all_contributions else 0,
            'mean_reward': np.mean(all_rewards) if all_rewards else 0,
            'top_contributors': sorted(self.total_rewards.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def detect_free_riders(self, threshold: float = 0.1) -> List[int]:
        """
        Detect potential free-riders (low contribution clients)
        """
        free_riders = []
        
        for client_id, history in self.contribution_history.items():
            if len(history) >= 5:
                avg_contrib = np.mean(history[-10:])
                if avg_contrib < threshold:
                    free_riders.append(client_id)
        
        return free_riders
    
    def compute_fairness_score(self) -> float:
        """
        Compute fairness of reward distribution using Gini coefficient
        """
        rewards = list(self.total_rewards.values())
        if not rewards:
            return 1.0
        
        rewards = np.array(rewards)
        n = len(rewards)
        
        # Gini coefficient
        sorted_rewards = np.sort(rewards)
        cumsum = np.cumsum(sorted_rewards)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # Convert to fairness (lower gini = more fair)
        fairness = 1.0 - gini
        
        return fairness