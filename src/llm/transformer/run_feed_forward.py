import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.feed_forward import PositionwiseFeedForward
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunFeedForward")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Position-wise Feed Forward Network...")
    
    # 1. Setup Configuration
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        d_ff=2048,
        activation_function="relu",
        dropout_rate=0.0  # Set to 0 for deterministic output
    )
    
    # 2. Instantiate Architecture
    ffn_module = PositionwiseFeedForward(config)
    
    # Verify parameter count
    num_params = count_parameters(ffn_module)
    params_w1 = config.d_model * config.d_ff + config.d_ff
    params_w2 = config.d_ff * config.d_model + config.d_model
    expected_params = params_w1 + params_w2
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Position-wise Feed Forward Network")
    logger.info(f"Input Dimension      : {config.d_model}")
    logger.info(f"Hidden Dimension     : {config.d_ff}")
    logger.info(f"Activation Function  : {config.activation_function.upper()}")
    logger.info(f"Total Trainable Params: {num_params:,} (Expected: {expected_params:,})")
    
    # 3. Create Dummy Tensors (representing output from Attention block)
    batch_size = 2
    seq_len = 10
    
    # Tensors shaped for FFN input: (batch_size, seq_len, d_model)
    # Using normal distribution to mimic normalized features
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    
    # 4. Forward Pass (Tracking activations for visualization)
    logger.info("Running forward pass...")
    
    # Intercept intermediate for visualization
    with torch.no_grad():
        x_expanded = ffn_module.w_1(hidden_states)
        x_act = ffn_module.activation(x_expanded)
        
        # Standard forward output
        output = ffn_module(hidden_states)
    
    logger.info("--- TENSOR SHAPES ---")
    logger.info(f"Input Vectors        : {hidden_states.shape} (batch_size, seq_len, d_model)")
    logger.info(f"Expanded Hidden      : {x_expanded.shape} (batch_size, seq_len, d_ff)")
    logger.info(f"Final Output         : {output.shape} (batch_size, seq_len, d_model)")
    
    # 5. Output Statistics
    logger.info("--- ACTIVATION STATISTICS ---")
    logger.info(f"Pre-Activation Mean  : {x_expanded.mean().item():.4f}")
    logger.info(f"Pre-Activation Std   : {x_expanded.std().item():.4f}")
    logger.info(f"Post-Activation Mean : {x_act.mean().item():.4f}")
    logger.info(f"Post-Activation Std  : {x_act.std().item():.4f}")
    
    # Note how ReLU kills the negative values, shifting the mean higher and reducing std.
    
    # Calculate Memory
    mem_mb = (output.element_size() * output.nelement()) / (1024 * 1024)
    logger.info(f"Output Matrix Memory Footprint: {mem_mb:.4f} MB")
    
    # 6. Visualization
    visualizer = RepresentationVisualizer(config)
    visualizer.plot_activation_distribution(x_expanded, x_act, "ffn_activation_distribution.png")
    
    elapsed = time.time() - t_start
    logger.info(f"Feed Forward pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
