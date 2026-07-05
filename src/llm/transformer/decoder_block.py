import torch
import torch.nn as nn
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.multi_head_attention import MultiHeadAttention
from src.llm.transformer.cross_attention import CrossAttention
from src.llm.transformer.feed_forward import PositionwiseFeedForward
from src.llm.transformer.normalization import SublayerConnection

class DecoderBlock(nn.Module):
    """
    A single Decoder block from "Attention Is All You Need".
    Consists of three primary sublayers:
    1. Masked Multi-Head Self-Attention (prevents attending to future tokens)
    2. Encoder-Decoder Cross Attention (focuses on the source sequence)
    3. Position-wise Feed Forward Network
    
    Each is wrapped in a SublayerConnection (LayerNorm + Residual Dropout).
    """
    def __init__(self, config: TransformerConfig):
        super(DecoderBlock, self).__init__()
        
        # Core mathematical components
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = CrossAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        
        # Sublayer wrappers
        self.sublayer_1 = SublayerConnection(config)
        self.sublayer_2 = SublayerConnection(config)
        self.sublayer_3 = SublayerConnection(config)

    def forward(
        self, 
        x: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder hidden states. Shape (batch_size, tgt_seq_len, d_model)
            encoder_hidden_states: Shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for self-attention (usually Causal Mask + Target Padding Mask)
            src_mask: Mask for cross-attention (Encoder Padding Mask)
            
        Returns:
            Processed tensor of exactly the same shape (batch_size, tgt_seq_len, d_model)
        """
        # 1. Masked Self-Attention
        # The target sequence attends to itself, strictly forbidding look-ahead
        self_attn_func = lambda inp: self.self_attention(query=inp, key=inp, value=inp, mask=tgt_mask)[0]
        x = self.sublayer_1(x, self_attn_func)
        
        # 2. Encoder-Decoder Cross Attention
        # Queries from Decoder, Keys/Values from Encoder
        cross_attn_func = lambda inp: self.cross_attention(decoder_hidden_states=inp, encoder_hidden_states=encoder_hidden_states, encoder_padding_mask=src_mask)[0]
        x = self.sublayer_2(x, cross_attn_func)
        
        # 3. Position-wise Feed Forward
        x = self.sublayer_3(x, self.feed_forward)
        
        return x
