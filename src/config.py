"""
Configuration settings for FedFIM
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import random
import numpy as np
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
CLIENT_MODELS_DIR = MODELS_DIR / "client_models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 SYNTHETIC_DATA_DIR, CLIENT_MODELS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data configuration"""
    # Market data
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "GOOGL", "MSFT", "AMZN", "META", 
        "TSLA", "NVDA", "JPM", "V", "JNJ"
    ])
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    
    # OPTIMIZED: Increased sequence length to capture longer trends
    sequence_length: int = 60
    prediction_horizon: int = 1
    
    # Features
    price_features: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'log_returns', 'volatility'
    ])
    technical_indicators: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'obv', 'cci'
    ])
    
    # Sentiment
    sentiment_sources: List[str] = field(default_factory=lambda: [
        'reddit', 'news', 'twitter'
    ])
    sentiment_embedding_dim: int = 384
    
    # Behavior
    behavior_features: List[str] = field(default_factory=lambda: [
        'portfolio_allocation', 'trade_frequency', 'holding_duration',
        'risk_tolerance', 'strategy_type', 'position_size'
    ])
    behavior_embedding_dim: int = 64

    # LLM Settings
    openrouter_model: str = field(default_factory=lambda: os.environ.get('OPENROUTER_MODEL', 'meta-llama/llama-3.3-70b-instruct:free'))


@dataclass
class ModelConfig:
    """Model configuration - BALANCED FOR REAL DATA"""
    # Price encoder
    price_input_dim: int = 22
    price_hidden_dim: int = 128  # Reduced from 256 to prevent overfitting
    price_num_layers: int = 2
    price_output_dim: int = 64   # Reduced from 128
    price_dropout: float = 0.3   # Increased from 0.2 for regularization
    
    # Sentiment encoder
    sentiment_input_dim: int = 384
    sentiment_hidden_dim: int = 128
    sentiment_output_dim: int = 64
    sentiment_dropout: float = 0.3
    
    # Behavior encoder
    behavior_input_dim: int = 10
    behavior_hidden_dim: int = 64
    behavior_output_dim: int = 32
    behavior_dropout: float = 0.3
    
    # Fusion
    fusion_hidden_dim: int = 128  # Reduced from 256
    fusion_output_dim: int = 64   # Reduced from 128
    
    # Heads
    num_classes_direction: int = 2  # Binary classification
    risk_output_dim: int = 1
    action_output_dim: int = 3
    
    # Personalization
    personalization_hidden_dim: int = 32
    adapter_hidden_dim: int = 32
    
    # Personalization
    personalization_hidden_dim: int = 32
    adapter_hidden_dim: int = 32


@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    num_clients: int = 20
    num_rounds: int = 50
    local_epochs: int = 5
    local_batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # FedProx
    mu: float = 0.01
    
    # Drift
    drift_threshold: float = 0.3
    drift_window: int = 5
    drift_adaptation_rate: float = 0.1
    
    # Incentive
    incentive_alpha: float = 0.3
    incentive_beta: float = 0.2
    incentive_gamma: float = 0.2
    incentive_delta: float = 0.3
    
    # Participation
    min_clients_per_round: int = 5
    client_selection_ratio: float = 0.8


@dataclass
class TrainingConfig:
    """Training configuration"""
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_interval: int = 10
    
    # Logging
    log_interval: int = 10
    tensorboard_log: bool = True


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Default configurations
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
FEDERATED_CONFIG = FederatedConfig()
TRAINING_CONFIG = TrainingConfig()