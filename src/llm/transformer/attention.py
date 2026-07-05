import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot Product Attention as defined in "Attention Is All You Need".
    Equation: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    This module contains NO trainable parameters.
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        # Note: Dropout during attention is optional but highly recommended to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, ..., seq_len_q, d_k)
            key: Tensor of shape (batch_size, ..., seq_len_k, d_k)
            value: Tensor of shape (batch_size, ..., seq_len_k, d_v)
            mask: Optional boolean Tensor of shape (batch_size, ..., seq_len_q, seq_len_k).
                  True indicates a valid position, False indicates a masked position.
                  
        Returns:
            output: Weighted values of shape (batch_size, ..., seq_len_q, d_v)
            p_attn: Attention probabilities of shape (batch_size, ..., seq_len_q, seq_len_k)
        """
        # Retrieve d_k from the last dimension of the query tensor
        d_k = query.size(-1)
        
        # 1. Dot product: Q * K^T
        # Transpose the last two dimensions of Key to allow matrix multiplication
        # query shape: (..., seq_len_q, d_k)
        # key.transpose(-2, -1) shape: (..., d_k, seq_len_k)
        # scores shape: (..., seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. Scale by 1 / sqrt(d_k)
        # Scaling prevents the dot products from growing too large and pushing the softmax 
        # function into regions where it has extremely small gradients.
        scores = scores / math.sqrt(d_k)
        
        # 3. Apply Mask (if provided)
        if mask is not None:
            # We fill the positions where mask == 0 (False) with a very large negative number
            # so that after softmax, their probability is essentially 0.
            # Using -1e9 instead of -float('inf') is standard to avoid NaN issues in fp16/bf16.
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Softmax
        # Compute probabilities along the last dimension (seq_len_k)
        p_attn = torch.softmax(scores, dim=-1)
        
        # 5. Optional Dropout
        p_attn = self.dropout(p_attn)
        
        # 6. Final weighted values: p_attn * V
        # p_attn shape: (..., seq_len_q, seq_len_k)
        # value shape: (..., seq_len_k, d_v)
        # output shape: (..., seq_len_q, d_v)
        output = torch.matmul(p_attn, value)
        
        return output, p_attn
