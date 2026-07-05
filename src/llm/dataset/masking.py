import torch

class MaskGenerator:
    """Generates attention and causal masks for Transformer training."""
    
    @staticmethod
    def create_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        Creates a boolean mask where True indicates a valid token and False indicates padding.
        Shape: (batch_size, seq_len)
        """
        # True for tokens that are NOT padding
        mask = (input_ids != pad_token_id)
        return mask

    @staticmethod
    def create_causal_mask(seq_len: int) -> torch.Tensor:
        """
        Creates a causal (look-ahead) mask for autoregressive generation.
        Returns a lower triangular matrix where 1s are allowed and 0s are masked out.
        Shape: (seq_len, seq_len)
        """
        # Creates a lower triangular matrix of ones.
        # e.g. for seq_len=3
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask
