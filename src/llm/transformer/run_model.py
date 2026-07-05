import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunModel")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Final CrackLaw Transformer Architecture...")
    
    # 1. Setup Configuration using standard paper sizes
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        tie_word_embeddings=False, # We explicitly turn this off to see max parameter count
        dropout_rate=0.0
    )
    
    # 2. Instantiate Architecture
    model = CrackLawTransformer(config)
    
    # Verify massive parameter count
    num_params = count_parameters(model)
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Full CrackLaw Transformer")
    logger.info(f"Total Trainable Params: {num_params:,}")
    
    # 3. Visualization mapping
    visualizer = RepresentationVisualizer(config)
    visualizer.generate_full_model_diagram("full_model_architecture.txt")
    
    # 4. Create Dummy Input
    batch_size = 2
    src_seq_len = 24
    tgt_seq_len = 16
    
    src_input_ids = torch.randint(1, config.vocab_size, (batch_size, src_seq_len))
    tgt_input_ids = torch.randint(1, config.vocab_size, (batch_size, tgt_seq_len))
    
    # 5. Full Teacher-Forced Forward Pass
    logger.info("Running complete end-to-end forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(
            src_input_ids=src_input_ids,
            tgt_input_ids=tgt_input_ids
        )
    
    logger.info("--- FINAL TENSOR SHAPES ---")
    logger.info(f"Source IDs          : {src_input_ids.shape}")
    logger.info(f"Target IDs          : {tgt_input_ids.shape}")
    logger.info(f"Output Logits       : {logits.shape}")
    
    # Memory profiling
    mem_mb = (logits.element_size() * logits.nelement()) / (1024 * 1024)
    logger.info(f"Massive Logits Matrix Memory Footprint: {mem_mb:.4f} MB")
    
    elapsed = time.time() - t_start
    logger.info(f"Transformer Assembly pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
