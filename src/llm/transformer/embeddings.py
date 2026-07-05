import torch
import torch.nn as nn
import math
from src.llm.transformer.config import TransformerConfig

class TokenEmbedding(nn.Module):
    """
    Learnable token embedding layer that converts token IDs into continuous vectors.
    """
    
    def __init__(self, config: TransformerConfig):
        super(TokenEmbedding, self).__init__()
        self.config = config
        
        # Core embedding table
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model
        )
        
        self._init_weights()

    def _init_weights(self):
        """
        Proper weight initialization. In standard Transformers, 
        embeddings are often scaled or initialized carefully.
        Here we use standard normal distribution with mean 0 and std 1.
        The scaling by sqrt(d_model) happens in the forward pass of InputRepresentation.
        """
        nn.init.normal_(self.embedding.weight, mean=0, std=self.config.d_model ** -0.5)
        # Assuming padding index is 0, we zero out the padding vector
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing token IDs.
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(x)
