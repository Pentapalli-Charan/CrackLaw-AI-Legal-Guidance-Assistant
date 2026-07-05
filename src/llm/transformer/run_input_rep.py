import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.input_representation import InputRepresentation
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunInputRepresentation")

def count_parameters(model: torch.nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Transformer Input Representation Module...")
    
    # 1. Setup Configuration
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        max_seq_len=1024,
        dropout_rate=0.1
    )
    
    # 2. Instantiate Architecture
    model = InputRepresentation(config)
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # 3. Print Architecture Report
    num_params = count_parameters(model)
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Vocabulary Size      : {config.vocab_size}")
    # Fix the reference below
    logger.info(f"Embedding Dimension  : {config.d_model}")
    logger.info(f"Max Sequence Length  : {config.max_seq_len}")
    logger.info(f"Total Trainable Params: {num_params:,}")
    
    # 4. Create Dummy Batch
    batch_size = 4
    seq_len = 128
    
    # Generate random token IDs mimicking a DataLoader batch
    # Values between 1 and vocab_size - 1 (0 reserved for pad)
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len)).to(device)
    
    # Set a few tokens to 0 to simulate padding
    input_ids[:, -5:] = 0
    
    # 5. Forward Pass
    logger.info("Running forward pass...")
    # Set to eval mode to disable dropout for deterministic testing
    model.eval() 
    with torch.no_grad():
        output_representation = model(input_ids)
        
    # 6. Generate Tensor Report & Visualizations
    visualizer = RepresentationVisualizer(config)
    visualizer.generate_tensor_report(input_ids, output_representation)
    
    # Plot the registered Positional Encoding buffer
    visualizer.plot_positional_encoding(model.positional_encoding.pe, "positional_encoding_heatmap.png")
    
    logger.info("--- EXAMPLE VECTORS ---")
    logger.info(f"Input IDs (Batch 0, first 5 tokens): {input_ids[0, :5].tolist()}")
    logger.info(f"Final Representation (Batch 0, Token 0, first 5 dims): {output_representation[0, 0, :5].tolist()}")
    
    # Check positional encoding logic specifically
    pe_vec = model.positional_encoding.pe[0, 0, :5]
    logger.info(f"Pure Positional Encoding (Pos 0, first 5 dims)    : {pe_vec.tolist()}")
    
    elapsed = time.time() - t_start
    logger.info(f"Input Representation pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
