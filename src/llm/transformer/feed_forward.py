import torch
import torch.nn as nn
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.activation import ActivationFactory
from src.llm.transformer.initialization import WeightInitializer

class PositionwiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed Forward Network (FFN) block.
    Equation: FFN(x) = max(0, xW1 + b1)W2 + b2 (Assuming ReLU activation)
    
    This operates on each position identically and independently.
    """
    
    def __init__(self, config: TransformerConfig):
        super(PositionwiseFeedForward, self).__init__()
        
        # 1. Linear Expansion layer (d_model -> d_ff)
        self.w_1 = nn.Linear(config.d_model, config.d_ff)
        
        # 2. Non-linear Activation
        self.activation = ActivationFactory.get_activation(config.activation_function)
        
        # 3. Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # 4. Linear Compression layer (d_ff -> d_model)
        self.w_2 = nn.Linear(config.d_ff, config.d_model)
        
        # Initialize weights
        WeightInitializer.init_feed_forward(self.w_1, config.activation_function)
        # The second layer typically follows a linear projection, Xavier is standard here
        nn.init.xavier_uniform_(self.w_2.weight)
        if self.w_2.bias is not None:
            nn.init.constant_(self.w_2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
               Note: because it's position-wise, it actually works on (..., d_model)
               
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Step 1: Expand dimension to d_ff
        x_expanded = self.w_1(x)
        
        # Step 2: Apply Non-linear Activation
        x_act = self.activation(x_expanded)
        
        # Step 3: Apply Dropout
        x_drop = self.dropout(x_act)
        
        # Step 4: Compress back to d_model
        output = self.w_2(x_drop)
        
        return output
