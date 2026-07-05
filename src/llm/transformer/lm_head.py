import torch
import torch.nn as nn
from typing import Optional
from src.llm.transformer.config import TransformerConfig

class LMHead(nn.Module):
    """
    Language Model Head.
    Projects the deep d_model hidden states from the Decoder back up to the Vocabulary Size
    to generate raw logits for next-token prediction.
    """
    def __init__(self, config: TransformerConfig, tied_weight: Optional[nn.Parameter] = None):
        super(LMHead, self).__init__()
        
        # We explicitly do not use bias by default in the LM head for modern transformers
        # (Though some implementations do, omitting it usually stabilizes tied embeddings)
        self.projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings and tied_weight is not None:
            # If weight tying is enabled, the projection weight shares the exact memory 
            # as the token embedding weight. This massively reduces parameter count
            # and helps regularize the model.
            self.projection.weight = tied_weight
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder hidden states. Shape (batch_size, seq_len, d_model)
        Returns:
            Vocabulary logits. Shape (batch_size, seq_len, vocab_size)
        """
        # Linear projection
        logits = self.projection(x)
        return logits
