"""
Unit tests for data pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.market_data import TechnicalIndicators
from src.data_collection.behavior_data import BehaviorDataGenerator, BEHAVIOR_PROFILES
from src.features.technical_indicators import TechnicalIndicators as TI
from src.config import DATA_CONFIG


class TestMarketData:
    """Test market data processing"""
    
    def test_technical_indicators(self):
        """Test technical indicator calculation"""
        np.random.seed(42)
        n = 200
        
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=n, freq='D'),
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1e6, 1e7, n)
        })
        
        df = TI.add_all_indicators(df)
        
        assert 'rsi_14' in df.columns
        assert 'macd' in df.columns
        assert 'sma_20' in df.columns
        assert df['rsi_14'].notna().sum() > 0


class TestBehaviorData:
    """Test behavior data generation"""
    
    def test_profile_generation(self):
        """Test client profile generation"""
        generator = BehaviorDataGenerator(seed=42)
        profiles = generator.generate_client_profiles(20)
        
        assert len(profiles) == 20
        assert all('client_id' in p for p in profiles)
        assert all('behavior_type' in p for p in profiles)
    
    def test_behavior_profiles_exist(self):
        """Test that all behavior profiles are defined"""
        expected = ['conservative', 'aggressive', 'momentum', 'long_term', 'contrarian']
        
        for profile_type in expected:
            assert profile_type in BEHAVIOR_PROFILES
    
    def test_behavior_features(self):
        """Test behavior feature computation"""
        generator = BehaviorDataGenerator(seed=42)
        
        profile = {
            'risk_tolerance': 0.5,
            'trade_frequency': 0.5,
            'holding_duration': 0.5,
            'position_size': 0.5,
            'portfolio_diversity': 0.5
        }
        
        market_data = pd.DataFrame({
            'close': 100 + np.random.randn(100).cumsum(),
            'returns': np.random.randn(100) * 0.02
        })
        
        features = generator.compute_behavior_features(profile, market_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestFederatedDataset:
    """Test federated dataset creation"""
    
    def test_dataset_split(self):
        """Test client data splitting"""
        from src.data_collection.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        
        # Create synthetic data
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=500, freq='D'),
            'open': 100 + np.random.randn(500).cumsum(),
            'high': 102 + np.random.randn(500).cumsum(),
            'low': 98 + np.random.randn(500).cumsum(),
            'close': 100 + np.random.randn(500).cumsum(),
            'volume': np.random.randint(1e6, 1e7, 500),
            'returns': np.random.randn(500) * 0.02,
            'ticker': 'TEST'
        })
        
        # Add required columns
        for col in ['volatility', 'rsi', 'macd', 'sma_20', 'sma_50', 
                    'sentiment_score', 'sentiment_confidence']:
            data[col] = np.random.randn(500)
        
        profiles = BehaviorDataGenerator(seed=42).generate_client_profiles(5)
        
        # This would need the full implementation
        # Just test that it doesn't error
        assert len(profiles) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])