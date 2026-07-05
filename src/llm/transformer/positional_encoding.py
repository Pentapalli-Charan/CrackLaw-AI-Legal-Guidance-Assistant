import torch
import torch.nn as nn
import math
from src.llm.transformer.config import TransformerConfig

class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of tokens in the sequence.
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, config: TransformerConfig):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.config = config
        
        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(config.max_seq_len, config.d_model)
        
        # Create a tensor representing positions (0 to max_seq_len - 1)
        # Shape: (max_seq_len, 1)
        position = torch.arange(0, config.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term (the denominator in the exponent)
        # Formula: 10000^(2i / d_model) = exp(2i * -log(10000) / d_model)
        # Step is 2 because sine is for even indices, cosine is for odd indices.
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model)
        )
        
        # Apply sine to even indices (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension (1, max_seq_len, d_model) for broadcasting during forward pass
        pe = pe.unsqueeze(0)
        
        # Register as a buffer. 
        # Buffers are non-learnable parameters that automatically move to the specified 
        # device (CPU/GPU) alongside the module's actual parameters.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encodings added to the input tensor.
            Shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Slicing self.pe to match the actual sequence length of the batch
        # self.pe is (1, max_seq_len, d_model), we take (1, seq_len, d_model)
        # Broadcasting automatically handles the batch_size dimension.
        return x + self.pe[:, :seq_len, :].requires_grad_(False)
