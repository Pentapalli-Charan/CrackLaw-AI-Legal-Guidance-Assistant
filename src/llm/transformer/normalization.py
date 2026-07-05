import torch
import torch.nn as nn
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.layer_norm import LayerNorm
from src.llm.transformer.residual import ResidualConnection
from typing import Callable

class SublayerConnection(nn.Module):
    """
    A unified wrapper that applies Layer Normalization and Residual Connections 
    around a sublayer (e.g., Multi-Head Attention or Feed Forward Network).
    
    Supports both:
    1. Post-Norm (Original Paper): Output = LayerNorm(x + Sublayer(x))
    2. Pre-Norm (Modern Variants): Output = x + Sublayer(LayerNorm(x))
    """
    def __init__(self, config: TransformerConfig):
        super(SublayerConnection, self).__init__()
        self.norm_first = config.norm_first
        
        self.layer_norm = LayerNorm(config)
        self.residual = ResidualConnection(config)

    def forward(self, x: torch.Tensor, sublayer_func: Callable) -> torch.Tensor:
        """
        Args:
            x: Input tensor. Shape: (batch_size, seq_len, d_model)
            sublayer_func: A callable representing the sublayer (MHA or FFN). 
                           It must take a single tensor input and return a single tensor output.
                           
        Returns:
            Processed tensor of the same shape.
        """
        if self.norm_first:
            # Pre-Norm: Normalize FIRST, then pass to sublayer, then add residual
            norm_x = self.layer_norm(x)
            sublayer_out = sublayer_func(norm_x)
            return self.residual(x, sublayer_out)
        else:
            # Post-Norm: Pass to sublayer, add residual, then Normalize LAST
            sublayer_out = sublayer_func(x)
            res_out = self.residual(x, sublayer_out)
            return self.layer_norm(res_out)
