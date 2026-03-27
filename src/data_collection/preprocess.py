"""
Data preprocessing and federated data splitting
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import DATA_CONFIG, FEDERATED_CONFIG, TRAINING_CONFIG
from src.data_collection.market_data import MarketDataCollector, TechnicalIndicators, MarketDataset
from src.data_collection.sentiment_data import SentimentProcessor
from src.data_collection.behavior_data import BehaviorDataGenerator


class FederatedDataset(Dataset):
    """PyTorch dataset for federated learning with multimodal features"""
    
    def __init__(self, features: Dict[str, np.ndarray], labels: np.ndarray,
                 client_id: int, sequence_length: int = 30):
        self.client_id = client_id
        self.sequence_length = sequence_length
        
        self.price_features = features['price']
        self.sentiment_features = features['sentiment']
        self.behavior_features = features['behavior']
        self.labels = labels
        
        self.n_samples = len(labels)
    
    def __len__(self):
        return max(0, self.n_samples - self.sequence_length)
    
    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length
        
        return {
            'price': torch.tensor(self.price_features[start_idx:end_idx], dtype=torch.float32),
            'sentiment': torch.tensor(self.sentiment_features[end_idx-1], dtype=torch.float32),
            'behavior': torch.tensor(self.behavior_features, dtype=torch.float32),
            'label': torch.tensor(self.labels[end_idx], dtype=torch.long),
            'client_id': torch.tensor(self.client_id, dtype=torch.long)
        }


class DataPreprocessor:
    """Preprocesses and splits data for federated learning"""
    
    def __init__(self, config=None):
        self.config = config or DATA_CONFIG
        self.market_collector = MarketDataCollector()
        
        # Check env var for LLM usage
        use_llm = os.environ.get('USE_LLM_SENTIMENT', 'false').lower() == 'true'
        self.sentiment_processor = SentimentProcessor()
        
        self.behavior_generator = BehaviorDataGenerator()
        self.scalers = {}
    
    def prepare_full_dataset(self, tickers: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Prepare full multimodal dataset"""
        tickers = tickers or self.config.tickers
        
        all_data = []
        all_profiles = self.behavior_generator.generate_client_profiles(FEDERATED_CONFIG.num_clients)
        
        for ticker in tickers:
            # Fetch market data
            market_data = self.market_collector.fetch_yahoo_finance(
                ticker, self.config.start_date, self.config.end_date
            )
            
            # Add technical indicators
            market_data = TechnicalIndicators.add_all_indicators(market_data)
            
            # Add sentiment features
            market_data = self.sentiment_processor.create_sentiment_features(market_data, ticker)
            
            market_data['ticker'] = ticker
            all_data.append(market_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        return combined_data, all_profiles
    
    def create_client_splits(self, data: pd.DataFrame, profiles: List[Dict],
                             non_iid: bool = True) -> Dict[int, FederatedDataset]:
        """Create non-IID federated data splits"""
        client_datasets = {}
        
        tickers = data['ticker'].unique()
        ticker_prefs = {}
        
        for profile in profiles:
            client_id = profile['client_id']
            n_preferences = np.random.randint(1, min(4, len(tickers) + 1))
            preferred_tickers = np.random.choice(tickers, size=n_preferences, replace=False)
            ticker_prefs[client_id] = preferred_tickers
        
        for profile in profiles:
            client_id = profile['client_id']
            preferred = ticker_prefs[client_id]
            
            client_data = data[data['ticker'].isin(preferred)].copy()
            
            if non_iid and len(client_data) > 10:
                behavior_type = profile['behavior_type']
                
                if behavior_type == 'aggressive':
                    frac = min(1.0, 0.8 + 0.4 * np.random.random())
                    client_data = client_data.sample(frac=frac)
                elif behavior_type == 'conservative':
                    client_data = client_data[client_data['volatility'] < client_data['volatility'].median()]
                elif behavior_type == 'momentum':
                    client_data = client_data[
                        (client_data['returns'] > 0.01) | 
                        (client_data['returns'] < -0.01)
                    ]
            
            if len(client_data) < self.config.sequence_length + 10:
                client_data = data.sample(n=min(len(data), 500))
            
            features = self._extract_features(client_data, profile)
            labels = self._create_labels(client_data, profile)
            
            if len(labels) > self.config.sequence_length:
                client_datasets[client_id] = FederatedDataset(
                    features, labels, client_id, self.config.sequence_length
                )
        
        return client_datasets
    
    def _extract_features(self, data: pd.DataFrame, profile: Dict) -> Dict[str, np.ndarray]:
        """Extract multimodal features"""
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'returns',
                      'log_returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                      'macd_hist', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                      'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv', 'cci']
        
        price_data = data[price_cols].values
        scaler = StandardScaler()
        price_features = scaler.fit_transform(price_data)
        self.scalers['price'] = scaler
        price_features = np.nan_to_num(price_features, 0)
        
        sentiment_cols = ['sentiment_score', 'sentiment_confidence', 
                         'sentiment_volatility', 'sentiment_ma_7', 
                         'sentiment_ma_14', 'sentiment_momentum']
        
        if all(col in data.columns for col in sentiment_cols):
            sentiment_data = data[sentiment_cols].values
        else:
            sentiment_data = np.random.randn(len(data), 384) * 0.1
        
        if sentiment_data.shape[1] < 384:
            padding = np.zeros((len(sentiment_data), 384 - sentiment_data.shape[1]))
            sentiment_data = np.hstack([sentiment_data, padding])
        
        sentiment_features = np.nan_to_num(sentiment_data, 0)
        
        behavior_features = self.behavior_generator.compute_behavior_features(profile, data)
        behavior_features = np.tile(behavior_features, (len(data), 1))
        
        return {
            'price': price_features,
            'sentiment': sentiment_features,
            'behavior': behavior_features
        }
    
    def _create_labels(self, data: pd.DataFrame, profile: Dict) -> np.ndarray:
        """
        Create labels for training - OPTIMIZED FOR BINARY CLASSIFICATION
        0 = Up
        1 = Down
        (Removed Neutral class to improve signal strength)
        """
        returns = data['returns'].values
        
        # Lower threshold for binary decision
        threshold = 0.005  # 0.5% movement
        
        labels = np.zeros(len(returns), dtype=np.int64)
        
        # Logic:
        # If return > 0.5% -> Class 0 (Up)
        # If return <= -0.5% -> Class 1 (Down)
        # Anything in between is assigned based on closest direction
        labels[returns > threshold] = 0   # up
        labels[returns <= -threshold] = 1  # down
        
        # Handle the "neutral" middle zone by assigning to nearest class
        # This forces the model to make a decision
        middle_mask = (returns > -threshold) & (returns <= threshold)
        labels[middle_mask] = (returns[middle_mask] > 0).astype(int) # 0 if positive, 1 if negative
        
        return labels
    
    def split_train_val_test(self, client_datasets: Dict[int, FederatedDataset],
                              train_ratio: float = 0.7, val_ratio: float = 0.15
                              ) -> Tuple[Dict, Dict, Dict]:
        """Split client data into train/val/test"""
        train_datasets = {}
        val_datasets = {}
        test_datasets = {}
        
        for client_id, dataset in client_datasets.items():
            n = len(dataset)
            train_size = int(n * train_ratio)
            val_size = int(n * val_ratio)
            
            indices = list(range(n))
            np.random.shuffle(indices)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            if train_idx:
                train_datasets[client_id] = self._create_subset(dataset, train_idx, client_id)
            if val_idx:
                val_datasets[client_id] = self._create_subset(dataset, val_idx, client_id)
            if test_idx:
                test_datasets[client_id] = self._create_subset(dataset, test_idx, client_id)
        
        return train_datasets, val_datasets, test_datasets
    
    def _create_subset(self, dataset: FederatedDataset, indices: List[int], 
                       client_id: int) -> FederatedDataset:
        """Create a subset of a federated dataset"""
        features = {
            'price': dataset.price_features[indices],
            'sentiment': dataset.sentiment_features[indices],
            'behavior': dataset.behavior_features[indices[:1]] if len(indices) > 0 else dataset.behavior_features
        }
        labels = dataset.labels[indices]
        
        return FederatedDataset(features, labels, client_id, dataset.sequence_length)


def create_data_loaders(client_datasets: Dict[int, FederatedDataset], 
                        batch_size: int = 32) -> Dict[int, DataLoader]:
    """Create data loaders for each client"""
    loaders = {}
    
    for client_id, dataset in client_datasets.items():
        if len(dataset) > 0:
            loaders[client_id] = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=False
            )
    
    return loaders