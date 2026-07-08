import torch.nn as nn

import math

class WeightInitializer:
    """
    Applies proper weight initialization strategies to PyTorch modules.
    Crucial for stabilizing the training of deep Transformer architectures.
    """
    
    @staticmethod
    def init_feed_forward(linear_layer: nn.Linear, activation_name: str = "relu", num_layers: int = 1):
        """
        Initializes a linear layer based on the activation function that follows it.
        Includes GPT-2 style depth scaling.
        """
        name = activation_name.lower().strip()
        
        # GPT-2 style scaled initialization for deeper layers
        std = 0.02
        if num_layers > 1:
            std = std / math.sqrt(2 * num_layers)

        if name == "relu":
            # Kaiming (He) Normal Initialization is optimal for ReLU
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
        elif name in ["gelu", "silu"]:
            # For GELU/SiLU, we often fall back to Xavier (Glorot) uniform or normal,
            # or a modified Kaiming initialization. Xavier is a solid baseline for smooth activations.
            nn.init.xavier_uniform_(linear_layer.weight)
        else:
            # Default fallback
            nn.init.xavier_uniform_(linear_layer.weight)
            
        # Initialize biases to 0
        if linear_layer.bias is not None:
            nn.init.constant_(linear_layer.bias, 0.0)
