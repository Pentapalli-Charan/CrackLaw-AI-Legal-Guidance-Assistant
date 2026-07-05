import torch
import torch.nn as nn
from src.llm.transformer.config import TransformerConfig

class LayerNorm(nn.Module):
    """
    Implements Layer Normalization natively.
    Normalizes the input across the last dimension (d_model).
    Equation: y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
    """
    def __init__(self, config: TransformerConfig):
        super(LayerNorm, self).__init__()
        self.eps = config.layer_norm_epsilon
        
        # Learnable parameters: Scale (gamma) and Shift (beta)
        # Initialized to ones and zeros respectively
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Normalized tensor of the exact same shape.
        """
        # 1. Compute Mean along the last dimension
        # Keep dimensions so we can broadcast the subtraction
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. Compute Variance along the last dimension
        # unbiased=False matches PyTorch's native LayerNorm default
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 3. Normalize
        # Add epsilon to prevent division by zero in case of zero variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. Scale and Shift using learnable parameters
        output = self.gamma * x_norm + self.beta
        
        return output
