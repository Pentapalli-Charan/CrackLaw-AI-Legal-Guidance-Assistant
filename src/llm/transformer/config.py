import os
from dataclasses import dataclass
from src.config import PROJECT_ROOT

@dataclass
class TransformerConfig:
    """Configuration for the CrackLaw Transformer Architecture."""
    
    # Vocabulary & Input Size
    vocab_size: int = 5000  # Default target vocab size for our BPE Tokenizer
    max_seq_len: int = 1024
    
    # Model Dimensions
    d_model: int = 512      # Standard dimension used in Attention Is All You Need
    num_heads: int = 8      # Number of attention heads
    d_ff: int = 2048        # Dimension of the feed-forward network
    
    # Structural Depths
    num_encoder_layers: int = 6 # Number of identical blocks in the encoder stack
    num_decoder_layers: int = 6 # Number of identical blocks in the decoder stack
    
    # Regularization
    dropout_rate: float = 0.1
    
    # Activation Function
    activation_function: str = "relu" # Options: relu, gelu, silu
    
    # Normalization
    layer_norm_epsilon: float = 1e-5
    norm_first: bool = False # False for Post-Norm (Original), True for Pre-Norm (Modern)
    
    # Model Architecture
    tie_word_embeddings: bool = False # Tying encoder, decoder, and LM head weights
    
    # Optional outputs
    visualization_dir: str = os.path.join(PROJECT_ROOT, "datasets", "visualizations")

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads
        
        valid_activations = ["relu", "gelu", "silu"]
        assert self.activation_function in valid_activations, f"activation_function must be one of {valid_activations}"
        
        os.makedirs(self.visualization_dir, exist_ok=True)
