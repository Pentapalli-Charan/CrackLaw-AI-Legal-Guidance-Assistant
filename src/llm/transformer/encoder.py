import torch
import torch.nn as nn
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.input_representation import InputRepresentation
from src.llm.transformer.encoder_stack import EncoderStack
from src.llm.transformer.layer_norm import LayerNorm

class Encoder(nn.Module):
    """
    The complete Transformer Encoder architecture.
    Composes:
    1. Input Representation (Token Embedding + Positional Encoding)
    2. N-Layer Encoder Stack
    3. Final Layer Normalization (Standard practice, especially when using Pre-Norm)
    """
    def __init__(self, config: TransformerConfig):
        super(Encoder, self).__init__()
        
        self.input_representation = InputRepresentation(config)
        self.encoder_stack = EncoderStack(config)
        self.final_layer_norm = LayerNorm(config)

    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full forward pass from raw token IDs to deeply encoded hidden states.
        
        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len)
            mask: Optional boolean mask for attention (batch_size, seq_len, seq_len)
            
        Returns:
            Encoded representation of shape (batch_size, seq_len, d_model)
        """
        # 1. Convert integer IDs to dense embeddings with positional encodings
        x = self.input_representation(input_ids)
        
        # 2. Process through the deep N-layer stack
        x = self.encoder_stack(x, mask)
        
        # 3. Final normalization
        x = self.final_layer_norm(x)
        
        return x
