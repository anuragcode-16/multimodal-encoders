"""
Neural network encoders for different modalities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

from src.config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PriceEncoderLSTM(nn.Module):
    """LSTM-based encoder for price time series"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.input_proj = nn.Linear(config.price_input_dim, config.price_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.price_hidden_dim,
            hidden_size=config.price_hidden_dim,
            num_layers=config.price_num_layers,
            batch_first=True,
            dropout=config.price_dropout if config.price_num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(config.price_hidden_dim)
        self.output_proj = nn.Linear(config.price_hidden_dim, config.price_output_dim)
        self.dropout = nn.Dropout(config.price_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, output_dim)
        """
        # Project input
        x = self.input_proj(x)
        x = F.gelu(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        h_last = lstm_out[:, -1, :]
        
        # Layer norm and projection
        h_last = self.layer_norm(h_last)
        output = self.output_proj(h_last)
        output = self.dropout(output)
        
        return output


class PriceEncoderTransformer(nn.Module):
    """Transformer-based encoder for price time series"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.input_proj = nn.Linear(config.price_input_dim, config.price_hidden_dim)
        self.pos_encoding = PositionalEncoding(config.price_hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.price_hidden_dim,
            nhead=4,
            dim_feedforward=config.price_hidden_dim * 4,
            dropout=config.price_dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(config.price_hidden_dim)
        self.output_proj = nn.Linear(config.price_hidden_dim, config.price_output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use mean pooling
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        
        output = self.output_proj(x)
        return output


class SentimentEncoder(nn.Module):
    """Encoder for sentiment embeddings"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.encoder = nn.Sequential(
            nn.Linear(config.sentiment_input_dim, config.sentiment_hidden_dim),
            nn.LayerNorm(config.sentiment_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.sentiment_dropout),
            
            nn.Linear(config.sentiment_hidden_dim, config.sentiment_hidden_dim),
            nn.LayerNorm(config.sentiment_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.sentiment_dropout),
            
            nn.Linear(config.sentiment_hidden_dim, config.sentiment_output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BehaviorEncoder(nn.Module):
    """Encoder for user behavior features"""
    
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        self.encoder = nn.Sequential(
            nn.Linear(config.behavior_input_dim, config.behavior_hidden_dim),
            nn.LayerNorm(config.behavior_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.behavior_dropout),
            
            nn.Linear(config.behavior_hidden_dim, config.behavior_hidden_dim),
            nn.LayerNorm(config.behavior_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.behavior_dropout),
            
            nn.Linear(config.behavior_hidden_dim, config.behavior_output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch dimension
        if x.dim() == 3:
            x = x[:, 0, :] if x.size(1) > 1 else x.squeeze(1)
        return self.encoder(x)


class MultimodalEncoder(nn.Module):
    """Combined multimodal encoder"""
    
    def __init__(self, use_transformer: bool = False, config=None):
        super().__init__()
        config = config or MODEL_CONFIG
        
        # Individual encoders
        if use_transformer:
            self.price_encoder = PriceEncoderTransformer(config)
        else:
            self.price_encoder = PriceEncoderLSTM(config)
        
        self.sentiment_encoder = SentimentEncoder(config)
        self.behavior_encoder = BehaviorEncoder(config)
        
        # Total embedding dimension
        total_dim = (config.price_output_dim + 
                    config.sentiment_output_dim + 
                    config.behavior_output_dim)
    
    def forward(self, price: torch.Tensor, sentiment: torch.Tensor, 
                behavior: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode all modalities
        """
        price_emb = self.price_encoder(price)
        sentiment_emb = self.sentiment_encoder(sentiment)
        behavior_emb = self.behavior_encoder(behavior)
        
        return {
            'price': price_emb,
            'sentiment': sentiment_emb,
            'behavior': behavior_emb,
            'combined': torch.cat([price_emb, sentiment_emb, behavior_emb], dim=-1)
        }