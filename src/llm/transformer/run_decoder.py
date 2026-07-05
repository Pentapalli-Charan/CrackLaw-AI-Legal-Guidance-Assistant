import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.decoder import Decoder
from src.llm.transformer.visualization import RepresentationVisualizer
from src.llm.transformer.masks import AttentionMasks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunDecoder")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Full Transformer Decoder Stack...")
    
    # 1. Setup Configuration using standard paper sizes
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_decoder_layers=6, # Standard 6-layer stack
        dropout_rate=0.0
    )
    
    # 2. Instantiate Architecture
    decoder = Decoder(config)
    
    # Verify massive parameter count
    num_params = count_parameters(decoder)
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Full Decoder Stack")
    logger.info(f"Layers (N)           : {config.num_decoder_layers}")
    logger.info(f"Total Trainable Params: {num_params:,}")
    
    # 3. Visualization mapping
    visualizer = RepresentationVisualizer(config)
    visualizer.generate_decoder_diagram("decoder_architecture.txt")
    
    # 4. Create Dummy Input
    batch_size = 2
    src_seq_len = 24
    tgt_seq_len = 16
    
    # Target IDs
    tgt_input_ids = torch.randint(1, config.vocab_size, (batch_size, tgt_seq_len))
    
    # Simulated Encoder output
    encoder_hidden_states = torch.randn(batch_size, src_seq_len, config.d_model)
    
    # Generate Causal Mask
    causal_mask = AttentionMasks.create_causal_mask(tgt_seq_len)
    
    # 5. Full Forward Pass
    logger.info("Running full Decoder forward pass...")
    # Using eval() to freeze dropouts for deterministic memory profiles
    decoder.eval()
    with torch.no_grad():
        decoded_hidden_states = decoder(
            tgt_input_ids=tgt_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            tgt_mask=causal_mask
        )
    
    logger.info("--- FINAL TENSOR SHAPES ---")
    logger.info(f"Target IDs          : {tgt_input_ids.shape}")
    logger.info(f"Encoder States      : {encoder_hidden_states.shape}")
    logger.info(f"Causal Mask         : {causal_mask.shape}")
    logger.info(f"Decoded Output      : {decoded_hidden_states.shape}")
    
    # Memory profiling
    mem_mb = (decoded_hidden_states.element_size() * decoded_hidden_states.nelement()) / (1024 * 1024)
    logger.info(f"Decoded Matrix Memory Footprint: {mem_mb:.4f} MB")
    
    elapsed = time.time() - t_start
    logger.info(f"Transformer Decoder pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
