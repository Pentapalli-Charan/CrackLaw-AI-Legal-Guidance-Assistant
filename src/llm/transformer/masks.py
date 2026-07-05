import torch

class AttentionMasks:
    """
    Utilities for generating masks compatible with Scaled Dot Product Attention.
    """

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Creates a causal (look-ahead) mask for autoregressive generation.
        Returns a lower triangular boolean matrix where True indicates valid positions.
        Shape: (seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask

    @staticmethod
    def combine_masks(padding_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Combines a batch padding mask with a causal mask.
        
        Args:
            padding_mask: Boolean tensor of shape (batch_size, seq_len). True = valid.
            causal_mask: Boolean tensor of shape (seq_len, seq_len). True = valid.
            
        Returns:
            Combined boolean mask of shape (batch_size, seq_len, seq_len). True = valid.
        """
        # padding_mask shape: (batch_size, seq_len)
        # We need to broadcast it to (batch_size, seq_len, seq_len)
        # Specifically, we want it to mask out the keys (the last dimension).
        # We reshape to (batch_size, 1, seq_len)
        expanded_padding_mask = padding_mask.unsqueeze(1)
        
        # Logical AND: A position is valid ONLY if it's valid in both the padding mask AND the causal mask.
        combined_mask = expanded_padding_mask & causal_mask
        
        return combined_mask
