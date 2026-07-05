import torch
import torch.nn as nn
import copy
from typing import Optional
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.decoder_block import DecoderBlock
from src.llm.transformer.encoder_stack import clone_module # Reuse cloning logic

class DecoderStack(nn.Module):
    """
    Dynamically stacks N identical Decoder blocks sequentially.
    """
    def __init__(self, config: TransformerConfig):
        super(DecoderStack, self).__init__()
        
        base_block = DecoderBlock(config)
        self.layers = clone_module(base_block, config.num_decoder_layers)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Passes the input sequentially through every decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_hidden_states: Tensor from Encoder (batch_size, src_seq_len, d_model)
            tgt_mask: Target sequence causal/padding mask
            src_mask: Source sequence padding mask
            
        Returns:
            The deeply decoded hidden states. Shape: (batch_size, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(
                x, 
                encoder_hidden_states=encoder_hidden_states,
                tgt_mask=tgt_mask,
                src_mask=src_mask
            )
            
        return x
