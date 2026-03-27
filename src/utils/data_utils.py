"""
Additional data utilities for alignment and processing
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def align_time_series(dfs: List[pd.DataFrame], 
                      date_col: str = 'date') -> pd.DataFrame:
    """
    Align multiple time series DataFrames on their date columns.
    Uses outer join to preserve all dates, fills missing with NaN.
    """
    if not dfs:
        return pd.DataFrame()
    
    # Ensure date column is datetime
    aligned = []
    for df in dfs:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        aligned.append(df)
    
    # Concatenate
    merged = pd.concat(aligned, axis=1).reset_index()
    return merged


def create_rolling_windows(data: np.ndarray, 
                           window_size: int, 
                           stride: int = 1) -> np.ndarray:
    """
    Create rolling window views of a 2D numpy array.
    
    Args:
        data: Input data of shape (n_samples, n_features)
        window_size: Size of the rolling window
        stride: Stride between windows
    
    Returns:
        Array of shape (n_windows, window_size, n_features)
    """
    n_samples, n_features = data.shape
    
    if n_samples < window_size:
        return np.array([])
    
    n_windows = (n_samples - window_size) // stride + 1
    shape = (n_windows, window_size, n_features)
    
    # Efficient stride tricks
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    
    return np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )


def add_temporal_features(dates: pd.Series) -> pd.DataFrame:
    """
    Add temporal features from a date series.
    
    Returns DataFrame with:
    - day_of_week
    - day_of_month
    - month
    - quarter
    - is_month_start
    - is_month_end
    """
    df = pd.DataFrame()
    
    dates = pd.to_datetime(dates)
    
    df['day_of_week'] = dates.dt.dayofweek
    df['day_of_month'] = dates.dt.day
    df['month'] = dates.dt.month
    df['quarter'] = dates.dt.quarter
    df['is_month_start'] = dates.dt.is_month_start.astype(int)
    df['is_month_end'] = dates.dt.is_month_end.astype(int)
    df['year'] = dates.dt.year
    
    # Cyclical encoding for periodic features
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def handle_missing_values(df: pd.DataFrame, 
                          method: str = 'ffill') -> pd.DataFrame:
    """
    Handle missing values in time series data.
    
    Methods:
    - 'ffill': Forward fill
    - 'bfill': Backward fill
    - 'interpolate': Linear interpolation
    - 'drop': Drop rows with missing values
    """
    df = df.copy()
    
    if method == 'ffill':
        df = df.ffill().bfill()  # bfill for leading NaNs
    elif method == 'bfill':
        df = df.bfill().ffill()
    elif method == 'interpolate':
        df = df.interpolate(method='linear').ffill().bfill()
    elif method == 'drop':
        df = df.dropna()
    
    return df


def detect_outliers_zscore(data: np.ndarray, 
                           threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Returns boolean mask where True indicates outlier.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    
    z_scores = np.abs((data - mean) / (std + 1e-10))
    
    return (z_scores > threshold).any(axis=1)


def calculate_support_resistance(prices: pd.Series, 
                                 window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate support and resistance levels using rolling min/max.
    """
    support = prices.rolling(window=window, center=True).min()
    resistance = prices.rolling(window=window, center=True).max()
    
    return support, resistance