import torch
from src.llm.transformer.config import TransformerConfig

class HeadReshaper:
    """
    Utility class for handling the mathematical tensor manipulations required to
    split and concatenate heads in Multi-Head Attention efficiently.
    """
    
    def __init__(self, config: TransformerConfig):
        self.num_heads = config.num_heads
        self.d_k = config.d_k
        self.d_v = config.d_v

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Splits a projected tensor into multiple heads.
        
        Args:
            tensor: Shape (batch_size, seq_len, d_model)
        Returns:
            Shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = tensor.size()
        
        # 1. Reshape to break d_model into (num_heads, d_k)
        # Result: (batch_size, seq_len, num_heads, d_k)
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 2. Transpose seq_len and num_heads so that attention operates across seq_len
        # Result: (batch_size, num_heads, seq_len, d_k)
        tensor = tensor.transpose(1, 2)
        
        return tensor

    def concat_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the split operation, concatenating all heads back into d_model.
        
        Args:
            tensor: Shape (batch_size, num_heads, seq_len, d_v)
        Returns:
            Shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_v = tensor.size()
        
        # 1. Transpose back to (batch_size, seq_len, num_heads, d_v)
        # contiguous() is required because transpose changes memory layout, 
        # and view() needs a contiguous block of memory.
        tensor = tensor.transpose(1, 2).contiguous()
        
        # 2. Reshape (flatten) the last two dimensions to reconstruct d_model
        # Result: (batch_size, seq_len, d_model)
        d_model = num_heads * d_v
        tensor = tensor.view(batch_size, seq_len, d_model)
        
        return tensor
