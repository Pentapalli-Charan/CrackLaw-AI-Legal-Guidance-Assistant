import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.encoder import Encoder
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunEncoder")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Full Transformer Encoder Stack...")
    
    # 1. Setup Configuration using standard paper sizes
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6, # Standard 6-layer stack
        dropout_rate=0.0
    )
    
    # 2. Instantiate Architecture
    encoder = Encoder(config)
    
    # Verify massive parameter count
    num_params = count_parameters(encoder)
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Full Encoder Stack")
    logger.info(f"Layers (N)           : {config.num_encoder_layers}")
    logger.info(f"Total Trainable Params: {num_params:,}")
    
    # 3. Visualization mapping
    visualizer = RepresentationVisualizer(config)
    visualizer.generate_encoder_diagram("encoder_architecture.txt")
    
    # 4. Create Dummy Input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    # Add dummy padding mask (True for valid, False for padding)
    # Let's say the last 2 tokens are padding
    mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, :, -2:] = False
    
    # 5. Full Forward Pass
    logger.info("Running full Encoder forward pass...")
    # Using eval() to freeze dropouts for deterministic memory profiles
    encoder.eval()
    with torch.no_grad():
        encoded_hidden_states = encoder(input_ids, mask=mask)
    
    logger.info("--- FINAL TENSOR SHAPES ---")
    logger.info(f"Input IDs           : {input_ids.shape}")
    logger.info(f"Attention Mask      : {mask.shape}")
    logger.info(f"Encoded Output      : {encoded_hidden_states.shape}")
    
    # Memory profiling
    mem_mb = (encoded_hidden_states.element_size() * encoded_hidden_states.nelement()) / (1024 * 1024)
    logger.info(f"Encoded Matrix Memory Footprint: {mem_mb:.4f} MB")
    
    elapsed = time.time() - t_start
    logger.info(f"Transformer Encoder pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
