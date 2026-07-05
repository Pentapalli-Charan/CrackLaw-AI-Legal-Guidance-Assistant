import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.encoder import Encoder
from src.llm.transformer.decoder import Decoder
from src.llm.transformer.lm_head import LMHead
from src.llm.transformer.masks import AttentionMasks

class Transformer(nn.Module):
    """
    The Core Transformer Architecture from 'Attention Is All You Need'.
    Strictly handles tensor routing between Encoder, Decoder, and LM Head.
    """
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Handle weight tying if configured
        tied_weight = None
        if config.tie_word_embeddings:
            # Tie Encoder and Decoder token embeddings
            self.decoder.input_representation.token_embedding.embedding.weight = \
                self.encoder.input_representation.token_embedding.embedding.weight
            # Provide this weight to the LM Head
            tied_weight = self.decoder.input_representation.token_embedding.embedding.weight
            
        self.lm_head = LMHead(config, tied_weight=tied_weight)

    def forward(
        self, 
        src_input_ids: torch.Tensor, 
        tgt_input_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src_input_ids: (batch_size, src_seq_len)
            tgt_input_ids: (batch_size, tgt_seq_len)
            src_mask: Encoder padding mask
            tgt_mask: Decoder causal + padding mask
            
        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size)
            encoder_hidden_states: (batch_size, src_seq_len, d_model)
            decoder_hidden_states: (batch_size, tgt_seq_len, d_model)
        """
        # 1. Encode Source Sequence
        encoder_hidden_states = self.encoder(src_input_ids, mask=src_mask)
        
        # 2. Decode Target Sequence (Conditioned on Encoder States)
        # Note: In training, tgt_input_ids is shifted right (Teacher Forcing).
        # We auto-generate the causal mask if not explicitly provided, though the higher-level 
        # model API will typically construct complete masks.
        if tgt_mask is None:
            tgt_seq_len = tgt_input_ids.size(1)
            tgt_mask = AttentionMasks.create_causal_mask(tgt_seq_len).to(tgt_input_ids.device)
            
        decoder_hidden_states = self.decoder(
            tgt_input_ids=tgt_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            tgt_mask=tgt_mask,
            src_mask=src_mask  # Cross-attention padding mask
        )
        
        # 3. Project to Vocabulary Space
        logits = self.lm_head(decoder_hidden_states)
        
        return logits, encoder_hidden_states, decoder_hidden_states
