"""
User behavior data generation and processing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random

# User behavior profiles
BEHAVIOR_PROFILES = {
    'conservative': {
        'risk_tolerance': 0.2,
        'trade_frequency': 0.3,  # Low frequency
        'holding_duration': 0.8,  # Long holding
        'position_size': 0.3,    # Small positions
        'portfolio_diversity': 0.9,  # High diversification
        'stop_loss_threshold': 0.05,
        'take_profit_threshold': 0.15,
        'market_sensitivity': 0.4,
        'news_reactivity': 0.6,
        'momentum_preference': 0.3,
    },
    'aggressive': {
        'risk_tolerance': 0.9,
        'trade_frequency': 0.8,  # High frequency
        'holding_duration': 0.2,  # Short holding
        'position_size': 0.8,    # Large positions
        'portfolio_diversity': 0.3,  # Low diversification
        'stop_loss_threshold': 0.15,
        'take_profit_threshold': 0.30,
        'market_sensitivity': 0.9,
        'news_reactivity': 0.7,
        'momentum_preference': 0.8,
    },
    'momentum': {
        'risk_tolerance': 0.6,
        'trade_frequency': 0.6,
        'holding_duration': 0.4,
        'position_size': 0.6,
        'portfolio_diversity': 0.5,
        'stop_loss_threshold': 0.08,
        'take_profit_threshold': 0.20,
        'market_sensitivity': 0.8,
        'news_reactivity': 0.8,
        'momentum_preference': 0.9,
    },
    'long_term': {
        'risk_tolerance': 0.4,
        'trade_frequency': 0.2,
        'holding_duration': 0.9,
        'position_size': 0.5,
        'portfolio_diversity': 0.7,
        'stop_loss_threshold': 0.10,
        'take_profit_threshold': 0.25,
        'market_sensitivity': 0.5,
        'news_reactivity': 0.4,
        'momentum_preference': 0.2,
    },
    'contrarian': {
        'risk_tolerance': 0.5,
        'trade_frequency': 0.5,
        'holding_duration': 0.5,
        'position_size': 0.5,
        'portfolio_diversity': 0.6,
        'stop_loss_threshold': 0.07,
        'take_profit_threshold': 0.18,
        'market_sensitivity': 0.7,
        'news_reactivity': 0.3,  # Counter-react to news
        'momentum_preference': 0.1,  # Contrarian
    }
}


class BehaviorDataGenerator:
    """Generates synthetic user behavior data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_client_profiles(self, num_clients: int) -> List[Dict]:
        """Generate behavior profiles for federated clients"""
        profiles = []
        profile_types = list(BEHAVIOR_PROFILES.keys())
        
        for client_id in range(num_clients):
            # Assign primary behavior type with some randomness
            primary_type = profile_types[client_id % len(profile_types)]
            base_profile = BEHAVIOR_PROFILES[primary_type].copy()
            
            # Add individual variation
            for key in base_profile:
                if key not in ['stop_loss_threshold', 'take_profit_threshold']:
                    noise = np.random.normal(0, 0.1)
                    base_profile[key] = np.clip(base_profile[key] + noise, 0, 1)
            
            # Client metadata
            base_profile['client_id'] = client_id
            base_profile['behavior_type'] = primary_type
            base_profile['data_quality'] = np.random.uniform(0.7, 1.0)
            base_profile['update_frequency'] = np.random.uniform(0.5, 1.0)
            
            profiles.append(base_profile)
        
        return profiles
    
    def generate_client_trades(self, profile: Dict, market_data: pd.DataFrame,
                                num_days: int = 365) -> pd.DataFrame:
        """Generate synthetic trading activity for a client"""
        dates = market_data['date'].values[:num_days]
        closes = market_data['close'].values[:num_days]
        
        trades = []
        position = 0
        cash = 100000
        portfolio_value = [cash]
        
        for i, (date, close) in enumerate(zip(dates, closes)):
            # Decision factors
            risk_tol = profile['risk_tolerance']
            trade_freq = profile['trade_frequency']
            momentum_pref = profile['momentum_preference']
            
            # Calculate momentum
            if i >= 5:
                returns = (closes[i] - closes[i-5]) / closes[i-5]
            else:
                returns = 0
            
            # Generate trading signals
            trade_probability = trade_freq * (1 + 0.3 * np.random.random())
            
            if np.random.random() < trade_probability:
                # Decide action
                if momentum_pref > 0.5:
                    # Momentum trading
                    if returns > 0.02:
                        action = 'buy'
                    elif returns < -0.02:
                        action = 'sell'
                    else:
                        action = 'hold'
                else:
                    # Contrarian
                    if returns < -0.02:
                        action = 'buy'
                    elif returns > 0.02:
                        action = 'sell'
                    else:
                        action = 'hold'
                
                # Risk-based sizing
                if action == 'buy' and position == 0:
                    position = int(cash * risk_tol / close)
                    cash -= position * close
                elif action == 'sell' and position > 0:
                    cash += position * close
                    position = 0
                
                trades.append({
                    'date': date,
                    'action': action,
                    'price': close,
                    'quantity': position if action == 'buy' else position,
                    'position': position,
                    'cash': cash,
                    'portfolio_value': cash + position * close
                })
            
            portfolio_value.append(cash + position * close)
        
        df_trades = pd.DataFrame(trades)
        df_trades['client_id'] = profile['client_id']
        df_trades['behavior_type'] = profile['behavior_type']
        
        return df_trades, portfolio_value
    
    def compute_behavior_features(self, profile: Dict, 
                                   market_data: pd.DataFrame) -> np.ndarray:
        """Compute behavior feature vector for a client"""
        # Extract key features
        features = [
            profile.get('risk_tolerance', 0.5),
            profile.get('trade_frequency', 0.5),
            profile.get('holding_duration', 0.5),
            profile.get('position_size', 0.5),
            profile.get('portfolio_diversity', 0.5),
            profile.get('stop_loss_threshold', 0.1),
            profile.get('take_profit_threshold', 0.2),
            profile.get('market_sensitivity', 0.5),
            profile.get('news_reactivity', 0.5),
            profile.get('momentum_preference', 0.5),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def generate_client_data_split(self, market_data: pd.DataFrame, 
                                    profiles: List[Dict],
                                    ticker_preferences: Dict[int, List[str]],
                                    non_iid_intensity: float = 0.5) -> Dict[int, pd.DataFrame]:
        """Generate non-IID data splits for federated learning"""
        client_data = {}
        
        for profile in profiles:
            client_id = profile['client_id']
            
            # Sample data based on behavior type
            behavior_type = profile['behavior_type']
            
            # Non-IID: different label distributions per client
            base_data = market_data.copy()
            
            # Add client-specific noise and shifts
            noise_level = non_iid_intensity * (1 - profile['data_quality'])
            
            # Add noise to features
            numeric_cols = base_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noise = np.random.normal(0, noise_level * base_data[col].std(), len(base_data))
                base_data[col] = base_data[col] + noise
            
            # Time-based split (different clients have different time periods)
            n = len(base_data)
            start_idx = np.random.randint(0, int(n * 0.2))
            end_idx = n - np.random.randint(0, int(n * 0.2))
            
            # Volume variation
            volume_factor = profile['trade_frequency']
            base_data['volume'] = base_data['volume'] * (0.5 + volume_factor)
            
            client_data[client_id] = base_data.iloc[start_idx:end_idx].reset_index(drop=True)
        
        return client_data


def create_behavior_labels(market_data: pd.DataFrame, profile: Dict) -> np.ndarray:
    """Create behavior-influenced labels"""
    # Base direction labels
    returns = market_data['returns'].values
    
    # Modify based on behavior
    risk_tol = profile['risk_tolerance']
    
    labels = np.zeros(len(returns), dtype=np.int64)
    
    # Adjust threshold based on risk tolerance
    threshold = 0.01 * (2 - risk_tol)  # Higher risk = lower threshold
    
    labels[returns > threshold] = 0  # up
    labels[returns < -threshold] = 1  # down
    labels[(returns >= -threshold) & (returns <= threshold)] = 2  # neutral
    
    return labels