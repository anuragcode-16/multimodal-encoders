"""
Market data collection using yfinance (Public Data Scraping)
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DATA_CONFIG, TRAINING_CONFIG


class MarketDataCollector:
    """
    Collects market data using the yfinance library.
    No API key is required.
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_yahoo_finance(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data using yfinance Ticker.history().
        """
        cache_file = self.cache_dir / f"{ticker}_yahoo.parquet"
        
        # Try to load from cache first
        if self.use_cache and cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        # Try Yahoo Finance
        try:
            print(f"Fetching real data for {ticker} via yfinance...")
            
            # Use Ticker object for more stable single-ticker fetching
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            if not df.empty:
                # Reset index to make 'Date' a column
                df = df.reset_index()
                
                # Standardize column names to lowercase
                # Map specific names if needed
                df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
                
                # Ensure we have standard columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Downloaded data missing required columns. Found: {df.columns}")

                # Save to cache
                if self.use_cache:
                    df.to_parquet(cache_file)
                
                print(f"Successfully fetched {len(df)} rows for {ticker}")
                return df
            else:
                raise ValueError("Downloaded empty dataframe")
                
        except Exception as e:
            print(f"Yahoo Finance fetch failed for {ticker}: {e}")
            print(f"Generating synthetic data for {ticker} as fallback...")
            return self._generate_synthetic_market_data(ticker, start_date, end_date)
    
    def _generate_synthetic_market_data(self, ticker: str, start_date: str, 
                                         end_date: str) -> pd.DataFrame:
        """Generate realistic synthetic market data"""
        np.random.seed(hash(ticker) % (2**32))
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='B')
        n = len(dates)
        
        # Base price varies by ticker
        base_price = 50 + (hash(ticker) % 450) + np.random.uniform(0, 100)
        
        # Generate price series
        returns = np.random.normal(0.0003, 0.02, n)
        
        # Add regime changes
        regime_points = np.random.choice(range(50, max(51, n-50)), size=min(2, max(0, n//50)), replace=False)
        for point in regime_points:
            if point < n:
                returns[point:] += np.random.choice([-0.001, 0.001])
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            'high': prices * (1 + np.abs(np.random.normal(0.005, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0.005, 0.01, n))),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e8, n).astype(int)
        })
        
        # Ensure high/low constraints
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        return df


class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # OBV
        direction = np.sign(df['close'].diff())
        direction.iloc[0] = 0
        df['obv'] = (direction * df['volume']).cumsum()
        
        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-10)
        
        # Drop NaN values
        df = df.dropna()
        
        return df


class MarketDataset(Dataset):
    """PyTorch dataset for market data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 30, 
                 prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'returns',
                       'log_returns', 'volatility', 'rsi', 'macd', 'macd_signal',
                       'macd_hist', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                       'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv', 'cci']
        
        self.features = data[feature_cols].values
        self.prices = data['close'].values
        
        # Normalize features
        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        # Handle any remaining NaN
        self.features = np.nan_to_num(self.features, 0)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        self.future_prices = []
        
        for i in range(len(self.features) - sequence_length - prediction_horizon):
            self.sequences.append(self.features[i:i+sequence_length])
            
            # Target: direction classification
            current_price = self.prices[i+sequence_length-1]
            future_price = self.prices[i+sequence_length+prediction_horizon-1]
            returns = (future_price - current_price) / current_price
            
            if returns > 0.01:
                target = 0  # up
            elif returns < -0.01:
                target = 1  # down
            else:
                target = 2  # neutral
            
            self.targets.append(target)
            self.future_prices.append(future_price)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.sequences[idx]),
            'target': torch.tensor(self.targets[idx]),
            'price': torch.tensor(self.future_prices[idx])
        }