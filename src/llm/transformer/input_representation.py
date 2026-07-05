import torch
import torch.nn as nn
import math
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.embeddings import TokenEmbedding
from src.llm.transformer.positional_encoding import SinusoidalPositionalEncoding

class InputRepresentation(nn.Module):
    """
    The complete input stage for the Transformer.
    Combines Token Embeddings, Positional Encodings, and Dropout.
    """
    
    def __init__(self, config: TransformerConfig):
        super(InputRepresentation, self).__init__()
        self.config = config
        
        self.token_embedding = TokenEmbedding(config)
        self.positional_encoding = SinusoidalPositionalEncoding(config)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) containing integer token IDs.
            
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) ready for the first Encoder block.
        """
        # 1. Lookup embeddings
        embeddings = self.token_embedding(x)
        
        # 2. Scale embeddings by sqrt(d_model) as per "Attention Is All You Need"
        # This keeps the variance of the embeddings similar to the positional encodings
        embeddings = embeddings * math.sqrt(self.config.d_model)
        
        # 3. Add positional encodings
        represented = self.positional_encoding(embeddings)
        
        # 4. Apply dropout
        output = self.dropout(represented)
        
        return output
