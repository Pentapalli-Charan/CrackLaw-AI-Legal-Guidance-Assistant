import torch
import torch.nn as nn
import copy
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.encoder_block import EncoderBlock

def clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Utility function to produce N identical (but independent) layers.
    Using deepcopy ensures that each layer gets its own independent set of learnable weights.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderStack(nn.Module):
    """
    Dynamically stacks N identical Encoder blocks sequentially.
    """
    def __init__(self, config: TransformerConfig):
        super(EncoderStack, self).__init__()
        
        # Instantiate a single block, then clone it N times to form the stack
        base_block = EncoderBlock(config)
        self.layers = clone_module(base_block, config.num_encoder_layers)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Passes the input sequentially through every layer in the stack.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask (usually padding mask for Encoder)
            
        Returns:
            The deeply encoded hidden states. Shape: (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
