import torch.nn as nn

class ActivationFactory:
    """
    Returns the appropriate PyTorch activation function based on configuration.
    """
    
    @staticmethod
    def get_activation(activation_name: str) -> nn.Module:
        """
        Args:
            activation_name (str): name of the activation function (relu, gelu, silu)
        Returns:
            nn.Module: The corresponding PyTorch activation function.
        """
        name = activation_name.lower().strip()
        
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            # SiLU (Sigmoid Linear Unit) is also known as Swish
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}. Supported: relu, gelu, silu.")
