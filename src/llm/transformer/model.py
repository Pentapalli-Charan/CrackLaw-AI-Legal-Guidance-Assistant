import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.transformer import Transformer
from src.llm.transformer.masks import AttentionMasks

class CrackLawTransformer(nn.Module):
    """
    The High-Level Public API for the CrackLaw Transformer.
    This module wraps the core routing logic and exposes explicit methods 
    for the Training Engine (forward) and Inference Engine (encode/decode).
    """
    def __init__(self, config: TransformerConfig):
        super(CrackLawTransformer, self).__init__()
        self.config = config
        self.model = Transformer(config)

    def forward(
        self, 
        src_input_ids: torch.Tensor, 
        tgt_input_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard Teacher-Forced Forward Pass (Used heavily during training).
        
        Args:
            src_input_ids: Integer IDs of the input text.
            tgt_input_ids: Integer IDs of the target text (usually shifted right by 1).
            src_padding_mask: Boolean mask indicating <PAD> tokens in source. (True for valid, False for padding).
            tgt_padding_mask: Boolean mask indicating <PAD> tokens in target. (True for valid, False for padding).
            
        Returns:
            Vocabulary logits. Shape: (batch_size, tgt_seq_len, vocab_size)
        """
        # Automatically handle the Causal Look-Ahead Mask logic
        tgt_seq_len = tgt_input_ids.size(1)
        causal_mask = AttentionMasks.create_causal_mask(tgt_seq_len).to(tgt_input_ids.device)
        
        if tgt_padding_mask is not None:
            # Combine causal mask with padding mask
            # causal_mask is (seq, seq), tgt_padding_mask is (batch, 1, 1, seq)
            tgt_mask = causal_mask & tgt_padding_mask
        else:
            tgt_mask = causal_mask
            
        logits, _, _ = self.model(
            src_input_ids=src_input_ids,
            tgt_input_ids=tgt_input_ids,
            src_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        return logits

    def encode(self, src_input_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference Helper: Runs ONLY the Encoder to compute and cache the context representations.
        """
        return self.model.encoder(src_input_ids, mask=src_padding_mask)

    def decode(
        self, 
        tgt_input_ids: torch.Tensor, 
        encoder_hidden_states: torch.Tensor, 
        src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Inference Helper: Runs ONLY the Decoder using the cached Encoder context.
        """
        tgt_seq_len = tgt_input_ids.size(1)
        causal_mask = AttentionMasks.create_causal_mask(tgt_seq_len).to(tgt_input_ids.device)
        
        return self.model.decoder(
            tgt_input_ids=tgt_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            tgt_mask=causal_mask,
            src_mask=src_padding_mask
        )

    def generate_logits(self, decoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Inference Helper: Projects Decoder outputs to Vocabulary space.
        """
        return self.model.lm_head(decoder_hidden_states)
