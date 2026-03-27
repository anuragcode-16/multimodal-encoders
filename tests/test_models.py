"""
Unit tests for FedFIM models
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.encoders import PriceEncoderLSTM, SentimentEncoder, BehaviorEncoder
from src.models.fusion import AttentionFusion, ConcatFusion
from src.models.heads import MultiTaskHead, PersonalizationHead
from src.models.fedfim import FedFIMModel, create_fedfim_model
from src.config import MODEL_CONFIG


class TestEncoders:
    """Test encoder modules"""
    
    def test_price_encoder_lstm(self):
        """Test LSTM price encoder"""
        encoder = PriceEncoderLSTM()
        
        batch_size = 16
        seq_len = 30
        input_dim = MODEL_CONFIG.price_input_dim
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = encoder(x)
        
        assert output.shape == (batch_size, MODEL_CONFIG.price_output_dim)
    
    def test_sentiment_encoder(self):
        """Test sentiment encoder"""
        encoder = SentimentEncoder()
        
        batch_size = 16
        x = torch.randn(batch_size, MODEL_CONFIG.sentiment_input_dim)
        output = encoder(x)
        
        assert output.shape == (batch_size, MODEL_CONFIG.sentiment_output_dim)
    
    def test_behavior_encoder(self):
        """Test behavior encoder"""
        encoder = BehaviorEncoder()
        
        batch_size = 16
        x = torch.randn(batch_size, MODEL_CONFIG.behavior_input_dim)
        output = encoder(x)
        
        assert output.shape == (batch_size, MODEL_CONFIG.behavior_output_dim)


class TestFusion:
    """Test fusion modules"""
    
    def test_attention_fusion(self):
        """Test attention-based fusion"""
        fusion = AttentionFusion()
        
        batch_size = 16
        embeddings = {
            'price': torch.randn(batch_size, MODEL_CONFIG.price_output_dim),
            'sentiment': torch.randn(batch_size, MODEL_CONFIG.sentiment_output_dim),
            'behavior': torch.randn(batch_size, MODEL_CONFIG.behavior_output_dim)
        }
        
        output = fusion(embeddings)
        assert output.shape == (batch_size, MODEL_CONFIG.fusion_output_dim)
    
    def test_concat_fusion(self):
        """Test concatenation fusion"""
        fusion = ConcatFusion()
        
        batch_size = 16
        embeddings = {
            'combined': torch.randn(batch_size, 
                                    MODEL_CONFIG.price_output_dim + 
                                    MODEL_CONFIG.sentiment_output_dim + 
                                    MODEL_CONFIG.behavior_output_dim)
        }
        
        output = fusion(embeddings)
        assert output.shape == (batch_size, MODEL_CONFIG.fusion_output_dim)


class TestHeads:
    """Test prediction heads"""
    
    def test_multi_task_head(self):
        """Test multi-task prediction head"""
        head = MultiTaskHead()
        
        batch_size = 16
        x = torch.randn(batch_size, MODEL_CONFIG.fusion_output_dim)
        outputs = head(x)
        
        assert 'direction' in outputs
        assert 'risk' in outputs
        assert 'action' in outputs
        
        assert outputs['direction'].shape == (batch_size, MODEL_CONFIG.num_classes_direction)
        assert outputs['risk'].shape == (batch_size,)
        assert outputs['action'].shape == (batch_size, MODEL_CONFIG.action_output_dim)
    
    def test_personalization_head(self):
        """Test personalization adapter"""
        head = PersonalizationHead()
        
        batch_size = 16
        x = torch.randn(batch_size, MODEL_CONFIG.fusion_output_dim)
        output = head(x)
        
        assert output.shape == (batch_size, MODEL_CONFIG.fusion_output_dim)


class TestFedFIMModel:
    """Test complete FedFIM model"""
    
    def test_model_creation(self):
        """Test model instantiation"""
        model = create_fedfim_model(num_clients=10, device='cpu')
        assert isinstance(model, FedFIMModel)
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = create_fedfim_model(num_clients=10, device='cpu')
        
        batch_size = 16
        seq_len = 30
        
        price = torch.randn(batch_size, seq_len, MODEL_CONFIG.price_input_dim)
        sentiment = torch.randn(batch_size, MODEL_CONFIG.sentiment_input_dim)
        behavior = torch.randn(batch_size, MODEL_CONFIG.behavior_input_dim)
        client_ids = torch.randint(0, 10, (batch_size,))
        
        outputs = model(price, sentiment, behavior, client_ids, use_personalization=True)
        
        assert 'direction' in outputs
        assert outputs['direction'].shape == (batch_size, MODEL_CONFIG.num_classes_direction)
    
    def test_global_vs_personalized(self):
        """Test global and personalized modes"""
        model = create_fedfim_model(num_clients=10, device='cpu')
        
        batch_size = 16
        seq_len = 30
        
        price = torch.randn(batch_size, seq_len, MODEL_CONFIG.price_input_dim)
        sentiment = torch.randn(batch_size, MODEL_CONFIG.sentiment_input_dim)
        behavior = torch.randn(batch_size, MODEL_CONFIG.behavior_input_dim)
        client_ids = torch.randint(0, 10, (batch_size,))
        
        # Global mode
        outputs_global = model(price, sentiment, behavior, client_ids, use_personalization=False)
        
        # Personalized mode
        outputs_personalized = model(price, sentiment, behavior, client_ids, use_personalization=True)
        
        # Outputs should be different
        assert not torch.allclose(outputs_global['direction'], outputs_personalized['direction'])
    
    def test_parameter_extraction(self):
        """Test global parameter extraction and setting"""
        model = create_fedfim_model(num_clients=10, device='cpu')
        
        # Get global parameters
        params = model.get_global_parameters()
        assert isinstance(params, dict)
        assert len(params) > 0
        
        # Set parameters
        model.set_global_parameters(params)
    
    def test_client_parameters(self):
        """Test client-specific parameter handling"""
        model = create_fedfim_model(num_clients=10, device='cpu')
        
        for client_id in range(10):
            params = model.get_client_parameters(client_id)
            assert isinstance(params, dict)
            
            model.set_client_parameters(client_id, params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])