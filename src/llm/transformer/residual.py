import torch
import torch.nn as nn
from src.llm.transformer.config import TransformerConfig

class ResidualConnection(nn.Module):
    """
    Implements a residual (skip) connection with optional dropout.
    Equation: Output = Input + Dropout(Sublayer_Output)
    
    This preserves the gradient flow directly from the final layers back to the first layers.
    """
    def __init__(self, config: TransformerConfig):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The original input tensor to the sublayer. Shape: (batch_size, seq_len, d_model)
            sublayer_output: The processed output from the sublayer. Shape: (batch_size, seq_len, d_model)
            
        Returns:
            The combined tensor of the same shape.
        """
        # Apply dropout to the sublayer's output before adding it back
        return x + self.dropout(sublayer_output)
