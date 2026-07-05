import torch
import torch.nn as nn
from src.llm.transformer.config import TransformerConfig

class AttentionProjections(nn.Module):
    """
    Houses the linear transformations for Multi-Head Attention.
    Projects the incoming Query, Key, and Value vectors into the combined multi-head space.
    """
    def __init__(self, config: TransformerConfig):
        super(AttentionProjections, self).__init__()
        
        # In a highly efficient implementation, we use a single linear layer
        # for all heads instead of instantiating N parallel layers.
        # This projects an input of size d_model to another vector of size d_model.
        # It will be split mathematically in the HeadReshaper.
        
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        
    def project_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Applies the initial Q, K, V linear projections."""
        return self.W_q(query), self.W_k(key), self.W_v(value)
        
    def project_output(self, concat_output: torch.Tensor):
        """Applies the final linear projection after heads are concatenated."""
        return self.W_o(concat_output)
