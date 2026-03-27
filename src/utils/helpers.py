"""
Helper utilities for FedFIM
"""
import numpy as np
import pandas as pd
import torch
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import pickle


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    """Save object to pickle file"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load object from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_ema(values: List[float], alpha: float = 0.3) -> List[float]:
    """Compute exponential moving average"""
    ema = [values[0]] if values else []
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """Normalize weights to sum to 1"""
    total = sum(weights.values())
    if total == 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """Compute confidence interval"""
    from scipy import stats
    arr = np.array(values)
    mean = np.mean(arr)
    sem = stats.sem(arr)
    interval = sem * stats.t.ppf((1 + confidence) / 2, len(arr) - 1)
    return mean - interval, mean + interval


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device() -> str:
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)