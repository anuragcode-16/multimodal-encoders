"""
Task-specific prediction heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.config import MODEL_CONFIG


class DirectionHead(nn.Module):
    """Classification head for price direction prediction"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim // 2, config.num_classes_direction)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RegressionHead(nn.Module):
    """Regression head for price prediction"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x).squeeze(-1)


class RiskHead(nn.Module):
    """Risk scoring head"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scorer(x).squeeze(-1)


class TradingActionHead(nn.Module):
    """Trading action prediction head"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(input_dim // 2, config.action_output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiTaskHead(nn.Module):
    """Combined multi-task prediction head"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        self.direction_head = DirectionHead(input_dim, config)
        self.regression_head = RegressionHead(input_dim, config)
        self.risk_head = RiskHead(input_dim, config)
        self.action_head = TradingActionHead(input_dim, config)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'direction': self.direction_head(x),
            'regression': self.regression_head(x),
            'risk': self.risk_head(x),
            'action': self.action_head(x)
        }


class PersonalizationHead(nn.Module):
    """Client-specific personalization adapter"""
    
    def __init__(self, input_dim: int = None, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        input_dim = input_dim or config.fusion_output_dim
        
        # Lightweight adapter layers
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, config.personalization_hidden_dim),
            nn.GELU(),
            nn.Linear(config.personalization_hidden_dim, input_dim)
        )
        
        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(x)
        return self.layer_norm(x + adapted)  # Residual connection