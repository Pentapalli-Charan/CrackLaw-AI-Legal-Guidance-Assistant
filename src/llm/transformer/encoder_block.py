import torch
import torch.nn as nn
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.multi_head_attention import MultiHeadAttention
from src.llm.transformer.feed_forward import PositionwiseFeedForward
from src.llm.transformer.normalization import SublayerConnection

class EncoderBlock(nn.Module):
    """
    A single Encoder block from "Attention Is All You Need".
    Consists of:
    1. Multi-Head Self-Attention (wrapped in a SublayerConnection with LayerNorm + Residual)
    2. Position-wise Feed Forward Network (wrapped in a SublayerConnection with LayerNorm + Residual)
    """
    def __init__(self, config: TransformerConfig):
        super(EncoderBlock, self).__init__()
        
        # Core components
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        
        # Sublayer connection wrappers (LayerNorm + Dropout + Residual)
        self.sublayer_1 = SublayerConnection(config)
        self.sublayer_2 = SublayerConnection(config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Processed tensor of exactly the same shape (batch_size, seq_len, d_model)
        """
        # 1. Self-Attention Sublayer
        # We define a lambda function to match the signature expected by SublayerConnection
        # In the Encoder, Self-Attention uses x as Query, Key, and Value identically
        attention_func = lambda inp: self.self_attention(query=inp, key=inp, value=inp, mask=mask)[0]
        
        x = self.sublayer_1(x, attention_func)
        
        # 2. Feed Forward Sublayer
        x = self.sublayer_2(x, self.feed_forward)
        
        return x
