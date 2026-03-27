"""
User behavior feature engineering
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class BehaviorFeatureEngineer:
    """
    Engineer user behavior features for personalization
    """
    
    def __init__(self):
        # Behavior profiles with feature templates
        self.profile_templates = {
            'conservative': {
                'risk_tolerance': 0.2,
                'trading_frequency': 0.3,
                'holding_period_preference': 0.8,
                'stop_loss_sensitivity': 0.7,
                'diversification_level': 0.9,
                'leverage_usage': 0.1
            },
            'aggressive': {
                'risk_tolerance': 0.9,
                'trading_frequency': 0.8,
                'holding_period_preference': 0.2,
                'stop_loss_sensitivity': 0.3,
                'diversification_level': 0.3,
                'leverage_usage': 0.9
            },
            'momentum': {
                'risk_tolerance': 0.6,
                'trading_frequency': 0.7,
                'holding_period_preference': 0.4,
                'stop_loss_sensitivity': 0.5,
                'diversification_level': 0.5,
                'leverage_usage': 0.6
            },
            'value': {
                'risk_tolerance': 0.4,
                'trading_frequency': 0.3,
                'holding_period_preference': 0.7,
                'stop_loss_sensitivity': 0.6,
                'diversification_level': 0.7,
                'leverage_usage': 0.2
            },
            'contrarian': {
                'risk_tolerance': 0.5,
                'trading_frequency': 0.5,
                'holding_period_preference': 0.5,
                'stop_loss_sensitivity': 0.5,
                'diversification_level': 0.6,
                'leverage_usage': 0.4
            }
        }
    
    def compute_behavior_features(self, trades: pd.DataFrame, 
                                  portfolio_history: pd.DataFrame = None) -> Dict:
        """
        Compute behavior features from trading history
        """
        features = {}
        
        if trades.empty:
            return self._get_default_features()
        
        # Trading frequency
        features['trading_frequency'] = self._compute_trading_frequency(trades)
        
        # Risk metrics
        features['risk_tolerance'] = self._compute_risk_tolerance(trades)
        features['position_concentration'] = self._compute_concentration(trades)
        
        # Holding behavior
        features['avg_holding_period'] = self._compute_avg_holding_period(trades)
        features['holding_period_preference'] = self._normalize_holding_preference(
            features['avg_holding_period']
        )
        
        # Performance behavior
        features['win_rate'] = self._compute_win_rate(trades)
        features['profit_factor'] = self._compute_profit_factor(trades)
        
        # Reaction patterns
        features['stop_loss_rate'] = self._compute_stop_loss_rate(trades)
        features['take_profit_rate'] = self._compute_take_profit_rate(trades)
        
        # Timing behavior
        features['market_on_close_preference'] = self._compute_moc_preference(trades)
        features['overnight_holding_rate'] = self._compute_overnight_rate(trades)
        
        # Diversification
        features['diversification_level'] = self._compute_diversification(trades)
        
        # Leverage (if applicable)
        features['leverage_usage'] = self._compute_leverage_usage(trades)
        
        # Behavior consistency
        features['behavior_consistency'] = self._compute_consistency(trades)
        
        return features
    
    def _compute_trading_frequency(self, trades: pd.DataFrame) -> float:
        if 'date' not in trades.columns:
            return 0.5
        trades_per_day = trades.groupby(trades['date'].dt.date).size().mean()
        return min(1.0, trades_per_day / 10)  # Normalize
    
    def _compute_risk_tolerance(self, trades: pd.DataFrame) -> float:
        if 'quantity' not in trades.columns or 'price' not in trades.columns:
            return 0.5
        position_values = trades['quantity'] * trades['price']
        if position_values.empty:
            return 0.5
        avg_position = position_values.mean()
        max_position = position_values.max()
        return min(1.0, avg_position / (max_position + 1e-10) * 2)
    
    def _compute_concentration(self, trades: pd.DataFrame) -> float:
        if 'ticker' not in trades.columns:
            return 0.5
        ticker_counts = trades['ticker'].value_counts()
        if ticker_counts.empty:
            return 0.5
        return 1 - (len(ticker_counts) / len(trades))
    
    def _compute_avg_holding_period(self, trades: pd.DataFrame) -> float:
        # Simplified - would need entry/exit tracking
        return 5.0  # Default 5 days
    
    def _normalize_holding_preference(self, avg_days: float) -> float:
        # 1 day = 0.0, 100+ days = 1.0
        return min(1.0, avg_days / 100)
    
    def _compute_win_rate(self, trades: pd.DataFrame) -> float:
        if 'pnl' not in trades.columns:
            return 0.5
        return (trades['pnl'] > 0).mean()
    
    def _compute_profit_factor(self, trades: pd.DataFrame) -> float:
        if 'pnl' not in trades.columns:
            return 1.0
        gains = trades[trades['pnl'] > 0]['pnl'].sum()
        losses = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        return gains / (losses + 1e-10)
    
    def _compute_stop_loss_rate(self, trades: pd.DataFrame) -> float:
        # Estimate stop loss usage from trade patterns
        return 0.3  # Default
    
    def _compute_take_profit_rate(self, trades: pd.DataFrame) -> float:
        return 0.3  # Default
    
    def _compute_moc_preference(self, trades: pd.DataFrame) -> float:
        return 0.5  # Default
    
    def _compute_overnight_rate(self, trades: pd.DataFrame) -> float:
        return 0.5  # Default
    
    def _compute_diversification(self, trades: pd.DataFrame) -> float:
        if 'ticker' not in trades.columns:
            return 0.5
        unique_tickers = trades['ticker'].nunique()
        return min(1.0, unique_tickers / 10)
    
    def _compute_leverage_usage(self, trades: pd.DataFrame) -> float:
        return 0.2  # Default low leverage
    
    def _compute_consistency(self, trades: pd.DataFrame) -> float:
        return 0.7  # Default
    
    def _get_default_features(self) -> Dict:
        return {key: 0.5 for key in [
            'trading_frequency', 'risk_tolerance', 'position_concentration',
            'avg_holding_period', 'holding_period_preference', 'win_rate',
            'profit_factor', 'stop_loss_rate', 'take_profit_rate',
            'market_on_close_preference', 'overnight_holding_rate',
            'diversification_level', 'leverage_usage', 'behavior_consistency'
        ]}
    
    def classify_behavior_type(self, features: Dict) -> str:
        """
        Classify user into behavior type based on features
        """
        scores = {}
        
        for profile_name, template in self.profile_templates.items():
            score = 0
            for feature, value in features.items():
                if feature in template:
                    # Negative distance (closer = higher score)
                    score -= abs(template[feature] - value)
            scores[profile_name] = score
        
        return max(scores, key=scores.get)
    
    def create_behavior_embedding(self, features: Dict, embedding_dim: int = 32) -> np.ndarray:
        """
        Create dense embedding from behavior features
        """
        feature_values = list(features.values())
        
        # Simple projection to embedding space
        np.random.seed(42)  # Reproducible projection
        projection = np.random.randn(len(feature_values), embedding_dim)
        embedding = np.dot(feature_values, projection)
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return embedding.astype(np.float32)
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        return [
            'trading_frequency', 'risk_tolerance', 'position_concentration',
            'avg_holding_period', 'holding_period_preference', 'win_rate',
            'profit_factor', 'stop_loss_rate', 'take_profit_rate',
            'market_on_close_preference', 'overnight_holding_rate',
            'diversification_level', 'leverage_usage', 'behavior_consistency'
        ]