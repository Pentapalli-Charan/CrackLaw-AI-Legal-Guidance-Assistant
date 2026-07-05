import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.multi_head_attention import MultiHeadAttention

class CrossAttention(nn.Module):
    """
    Implements Encoder-Decoder Attention.
    Semantically wraps Multi-Head Attention to clearly define the input routing:
    - Query: Comes from the Decoder's current sequence.
    - Key & Value: Come from the Encoder's final hidden states.
    """
    def __init__(self, config: TransformerConfig):
        super(CrossAttention, self).__init__()
        # We completely reuse our mathematically proven MHA engine
        self.mha = MultiHeadAttention(config)

    def forward(
        self, 
        decoder_hidden_states: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden_states: Shape (batch_size, tgt_seq_len, d_model)
            encoder_hidden_states: Shape (batch_size, src_seq_len, d_model)
            encoder_padding_mask: Shape (batch_size, 1, tgt_seq_len, src_seq_len) or similar broadcastable boolean mask.
                                  Prevents the decoder from attending to `<PAD>` tokens in the encoder input.
                                  
        Returns:
            output: Attended representation of shape (batch_size, tgt_seq_len, d_model)
            attention_probs: The raw probability matrix of shape (batch_size, num_heads, tgt_seq_len, src_seq_len)
        """
        output, p_attn = self.mha(
            query=decoder_hidden_states, 
            key=encoder_hidden_states, 
            value=encoder_hidden_states, 
            mask=encoder_padding_mask
        )
        return output, p_attn
