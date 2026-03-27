"""
Multimodal fusion modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from src.config import MODEL_CONFIG


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with MLP"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        input_dim = (config.price_output_dim + 
                    config.sentiment_output_dim + 
                    config.behavior_output_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(config.fusion_hidden_dim, config.fusion_output_dim)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined = embeddings['combined']
        return self.fusion(combined)


class AttentionFusion(nn.Module):
    """Attention-based multimodal fusion"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.price_dim = config.price_output_dim
        self.sentiment_dim = config.sentiment_output_dim
        self.behavior_dim = config.behavior_output_dim
        
        # Project all to same dimension
        hidden_dim = max(self.price_dim, self.sentiment_dim, self.behavior_dim)
        
        self.price_proj = nn.Linear(self.price_dim, hidden_dim)
        self.sentiment_proj = nn.Linear(self.sentiment_dim, hidden_dim)
        self.behavior_proj = nn.Linear(self.behavior_dim, hidden_dim)
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 3),  # 3 modalities
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.fusion_hidden_dim, config.fusion_output_dim)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Project modalities
        price = self.price_proj(embeddings['price'])
        sentiment = self.sentiment_proj(embeddings['sentiment'])
        behavior = self.behavior_proj(embeddings['behavior'])
        
        # Stack modalities
        stacked = torch.stack([price, sentiment, behavior], dim=1)  # (batch, 3, hidden)
        
        # Compute attention weights
        attn_input = stacked.mean(dim=1)  # (batch, hidden)
        attn_weights = self.attention(attn_input)  # (batch, 3)
        
        # Weighted sum
        weighted = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return self.output_proj(weighted)


class GatedFusion(nn.Module):
    """Gated multimodal fusion"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.price_dim = config.price_output_dim
        self.sentiment_dim = config.sentiment_output_dim
        self.behavior_dim = config.behavior_output_dim
        
        hidden_dim = config.fusion_hidden_dim
        
        # Gates for each modality
        self.price_gate = nn.Sequential(
            nn.Linear(self.price_dim + self.sentiment_dim + self.behavior_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.sentiment_gate = nn.Sequential(
            nn.Linear(self.price_dim + self.sentiment_dim + self.behavior_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.behavior_gate = nn.Sequential(
            nn.Linear(self.price_dim + self.sentiment_dim + self.behavior_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Projections
        self.price_proj = nn.Linear(self.price_dim, hidden_dim)
        self.sentiment_proj = nn.Linear(self.sentiment_dim, hidden_dim)
        self.behavior_proj = nn.Linear(self.behavior_dim, hidden_dim)
        
        # Output
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, config.fusion_output_dim)
        )
    
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all embeddings for gate computation
        concat = torch.cat([
            embeddings['price'],
            embeddings['sentiment'],
            embeddings['behavior']
        ], dim=-1)
        
        # Compute gates
        price_gate = self.price_gate(concat)
        sentiment_gate = self.sentiment_gate(concat)
        behavior_gate = self.behavior_gate(concat)
        
        # Project and gate
        price_gated = self.price_proj(embeddings['price']) * price_gate
        sentiment_gated = self.sentiment_proj(embeddings['sentiment']) * sentiment_gate
        behavior_gated = self.behavior_proj(embeddings['behavior']) * behavior_gate
        
        # Sum gated features
        fused = price_gated + sentiment_gated + behavior_gated
        
        return self.output(fused)