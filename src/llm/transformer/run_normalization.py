import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.layer_norm import LayerNorm
from src.llm.transformer.normalization import SublayerConnection
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunNormalization")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Normalization & Residual Modules...")
    
    # 1. Setup Configuration
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        dropout_rate=0.0, # 0 for deterministic output
        norm_first=False  # Post-Norm
    )
    
    # 2. Instantiate Architecture
    layer_norm = LayerNorm(config)
    sublayer_conn = SublayerConnection(config)
    
    # Verify parameter count
    num_params = count_parameters(layer_norm)
    expected_params = config.d_model * 2 # gamma and beta
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Layer Normalization")
    logger.info(f"Dimensions (d_model) : {config.d_model}")
    logger.info(f"Architecture Type    : {'Pre-Norm' if config.norm_first else 'Post-Norm'}")
    logger.info(f"Total Trainable Params: {num_params:,} (Expected: {expected_params:,})")
    
    # 3. Create Dummy Tensors
    batch_size = 2
    seq_len = 10
    
    # Intentionally skew the distribution: Mean ~25, Std ~10
    input_tensor = torch.randn(batch_size, seq_len, config.d_model) * 10.0 + 25.0
    
    # 4. Forward Pass (LayerNorm directly for analysis)
    logger.info("Running LayerNorm forward pass...")
    output_norm = layer_norm(input_tensor)
    
    logger.info("--- TENSOR SHAPES ---")
    logger.info(f"Input Vectors        : {input_tensor.shape}")
    logger.info(f"Normalized Output    : {output_norm.shape}")
    
    # 5. Output Statistics
    logger.info("--- NORMALIZATION STATISTICS ---")
    logger.info("Focusing on Batch 0, Token 0 across all 512 dimensions:")
    pre_mean = input_tensor[0, 0].mean().item()
    pre_std = input_tensor[0, 0].std(unbiased=False).item()
    post_mean = output_norm[0, 0].mean().item()
    post_std = output_norm[0, 0].std(unbiased=False).item()
    
    logger.info(f"Pre-Norm Mean  : {pre_mean:.4f}")
    logger.info(f"Pre-Norm Std   : {pre_std:.4f}")
    logger.info(f"Post-Norm Mean : {post_mean:.4f}")
    logger.info(f"Post-Norm Std  : {post_std:.4f}")
    
    # 6. Run Sublayer Connection (Residual)
    logger.info("--- RESIDUAL SUBLAYER TEST ---")
    # A dummy sublayer that just adds 5 to everything
    def dummy_sublayer(x):
        return x + 5.0
        
    sub_output = sublayer_conn(input_tensor, dummy_sublayer)
    logger.info(f"Sublayer Wrapper Output Shape: {sub_output.shape}")
    
    # 7. Visualization
    visualizer = RepresentationVisualizer(config)
    visualizer.plot_normalization_statistics(input_tensor, output_norm, "layer_norm_statistics.png")
    
    elapsed = time.time() - t_start
    logger.info(f"Normalization pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
