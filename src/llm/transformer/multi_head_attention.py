import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.attention import ScaledDotProductAttention
from src.llm.transformer.projections import AttentionProjections
from src.llm.transformer.head import HeadReshaper

class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Attention from the "Attention Is All You Need" paper.
    Composes the Scaled Dot Product Attention mechanism with parallel multi-head projections.
    """
    def __init__(self, config: TransformerConfig):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        
        # Linear layers for projecting Q, K, V and the final output
        self.projections = AttentionProjections(config)
        
        # Utility for efficient mathematical reshaping of heads
        self.reshaper = HeadReshaper(config)
        
        # The core mathematical attention engine
        self.attention_engine = ScaledDotProductAttention(dropout_rate=config.dropout_rate)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: Input tensors, typically of shape (batch_size, seq_len, d_model)
            mask: Optional broadcastable boolean mask (e.g., shape (batch_size, 1, seq_len_q, seq_len_k) 
                  or (batch_size, seq_len_q, seq_len_k))
                  
        Returns:
            output: The attended and projected output of shape (batch_size, seq_len_q, d_model)
            p_attn: The raw attention probabilities (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 1. Linear Projections
        # Pass inputs through learnable weight matrices W_q, W_k, W_v
        # Output shapes: (batch_size, seq_len, d_model)
        proj_q, proj_k, proj_v = self.projections.project_qkv(query, key, value)
        
        # 2. Split Heads
        # Efficiently reshape into (batch_size, num_heads, seq_len, d_k)
        q_heads = self.reshaper.split_heads(proj_q)
        k_heads = self.reshaper.split_heads(proj_k)
        v_heads = self.reshaper.split_heads(proj_v)
        
        # Ensure mask broadcasts across the num_heads dimension
        # If mask is (batch_size, seq_len, seq_len), we need to add the heads dimension -> (batch_size, 1, seq_len, seq_len)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # 3. Scaled Dot Product Attention
        # Run the core attention engine independently and in parallel across all heads
        # out_heads: (batch_size, num_heads, seq_len, d_v)
        # p_attn: (batch_size, num_heads, seq_len, seq_len)
        out_heads, p_attn = self.attention_engine(q_heads, k_heads, v_heads, mask=mask)
        
        # 4. Concatenate Heads
        # Reconstruct the d_model dimension: (batch_size, seq_len, d_model)
        concat_out = self.reshaper.concat_heads(out_heads)
        
        # 5. Final Output Projection
        # Pass through the final weight matrix W_o
        # Output shape: (batch_size, seq_len, d_model)
        final_output = self.projections.project_output(concat_out)
        
        return final_output, p_attn
