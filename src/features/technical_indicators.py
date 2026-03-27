"""
Technical indicator feature engineering
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


class TechnicalIndicators:
    """
    Calculate technical indicators for financial time series
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, 
                           open_col: str = 'open',
                           high_col: str = 'high', 
                           low_col: str = 'low',
                           close_col: str = 'close',
                           volume_col: str = 'volume') -> pd.DataFrame:
        """
        Add all technical indicators to dataframe
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df[close_col].pct_change()
        df['log_returns'] = np.log(df[close_col] / df[close_col].shift(1))
        
        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df[close_col].rolling(period).mean()
            df[f'ema_{period}'] = df[close_col].ewm(span=period).mean()
        
        # Moving Average Convergence Divergence (MACD)
        df['macd'] = df[close_col].ewm(span=12).mean() - df[close_col].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Relative Strength Index (RSI)
        df['rsi_14'] = TechnicalIndicators._calculate_rsi(df[close_col], 14)
        df['rsi_7'] = TechnicalIndicators._calculate_rsi(df[close_col], 7)
        
        # Bollinger Bands
        df['bb_middle'] = df[close_col].rolling(20).mean()
        bb_std = df[close_col].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range (ATR)
        df['atr_14'] = TechnicalIndicators._calculate_atr(df[high_col], df[low_col], df[close_col], 14)
        
        # On-Balance Volume (OBV)
        df['obv'] = TechnicalIndicators._calculate_obv(df[close_col], df[volume_col])
        
        # Commodity Channel Index (CCI)
        df['cci_20'] = TechnicalIndicators._calculate_cci(df[high_col], df[low_col], df[close_col], 20)
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = TechnicalIndicators._calculate_stochastic(
            df[high_col], df[low_col], df[close_col]
        )
        
        # Average Directional Index (ADX)
        df['adx'] = TechnicalIndicators._calculate_adx(df[high_col], df[low_col], df[close_col])
        
        # Rate of Change (ROC)
        df['roc_10'] = TechnicalIndicators._calculate_roc(df[close_col], 10)
        df['roc_20'] = TechnicalIndicators._calculate_roc(df[close_col], 20)
        
        # Momentum
        df['momentum_10'] = df[close_col] - df[close_col].shift(10)
        df['momentum_20'] = df[close_col] - df[close_col].shift(20)
        
        # Williams %R
        df['williams_r'] = TechnicalIndicators._calculate_williams_r(
            df[high_col], df[low_col], df[close_col]
        )
        
        # Volume indicators
        df['volume_sma_20'] = df[volume_col].rolling(20).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_sma_20']
        
        # Price patterns
        df['higher_high'] = (df[high_col] > df[high_col].shift(1)).astype(int)
        df['lower_low'] = (df[low_col] < df[low_col].shift(1)).astype(int)
        
        # Candle patterns
        df['body_size'] = abs(df[close_col] - df[open_col])
        df['upper_shadow'] = df[high_col] - df[[open_col, close_col]].max(axis=1)
        df['lower_shadow'] = df[[open_col, close_col]].min(axis=1) - df[low_col]
        df['doji'] = (df['body_size'] < (df[high_col] - df[low_col]) * 0.1).astype(int)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                       period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff())
        direction.iloc[0] = 0
        return (direction * volume).cumsum()
    
    @staticmethod
    def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 20) -> pd.Series:
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-10)
    
    @staticmethod
    def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                              k_period: int = 14, d_period: int = 3):
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()
        return stoch_k, stoch_d
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 14) -> pd.Series:
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = TechnicalIndicators._calculate_atr(high, low, close, 1)
        plus_di = 100 * (plus_dm.rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        minus_di = 100 * (abs(minus_dm).rolling(period).mean() / (tr.rolling(period).mean() + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx
    
    @staticmethod
    def _calculate_roc(prices: pd.Series, period: int) -> pd.Series:
        return 100 * (prices - prices.shift(period)) / (prices.shift(period) + 1e-10)
    
    @staticmethod
    def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                              period: int = 14) -> pd.Series:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """Return list of all generated feature column names"""
        return [
            'returns', 'log_returns',
            'volatility_10', 'volatility_20', 'volatility_60',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14', 'rsi_7',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
            'atr_14', 'obv', 'cci_20',
            'stoch_k', 'stoch_d', 'adx',
            'roc_10', 'roc_20',
            'momentum_10', 'momentum_20',
            'williams_r',
            'volume_sma_20', 'volume_ratio',
            'higher_high', 'lower_low',
            'body_size', 'upper_shadow', 'lower_shadow', 'doji'
        ]