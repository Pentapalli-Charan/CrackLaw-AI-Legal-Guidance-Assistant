import torch
import torch.nn as nn
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.input_representation import InputRepresentation
from src.llm.transformer.decoder_stack import DecoderStack
from src.llm.transformer.layer_norm import LayerNorm

class Decoder(nn.Module):
    """
    The complete Transformer Decoder architecture.
    Composes:
    1. Input Representation (Target Token Embedding + Positional Encoding)
    2. N-Layer Decoder Stack (Masked Self-Attention + Cross-Attention + FFN)
    3. Final Layer Normalization
    """
    def __init__(self, config: TransformerConfig):
        super(Decoder, self).__init__()
        
        self.input_representation = InputRepresentation(config)
        self.decoder_stack = DecoderStack(config)
        self.final_layer_norm = LayerNorm(config)

    def forward(
        self, 
        tgt_input_ids: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass from raw target token IDs to fully decoded context states.
        
        Args:
            tgt_input_ids: Integer tensor of shape (batch_size, tgt_seq_len)
            encoder_hidden_states: Tensor from Encoder (batch_size, src_seq_len, d_model)
            tgt_mask: Causal Mask (+ target padding mask) for Self-Attention
            src_mask: Encoder padding mask for Cross-Attention
            
        Returns:
            Decoded representation of shape (batch_size, tgt_seq_len, d_model)
        """
        # 1. Convert integer IDs to dense embeddings with positional encodings
        x = self.input_representation(tgt_input_ids)
        
        # 2. Process through the deep N-layer decoder stack, conditioning on the Encoder's output
        x = self.decoder_stack(
            x, 
            encoder_hidden_states=encoder_hidden_states,
            tgt_mask=tgt_mask,
            src_mask=src_mask
        )
        
        # 3. Final normalization
        x = self.final_layer_norm(x)
        
        return x
