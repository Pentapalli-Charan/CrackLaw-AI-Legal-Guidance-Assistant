import os
import torch
import logging
import time
from src.llm.transformer.attention import ScaledDotProductAttention
from src.llm.transformer.masks import AttentionMasks
from src.llm.transformer.visualization import RepresentationVisualizer
from src.llm.transformer.config import TransformerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunAttention")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Scaled Dot Product Attention Module...")
    
    # 1. Configuration setup (dummy just for visualization dir)
    config = TransformerConfig()
    
    # 2. Instantiate Architecture
    attention_module = ScaledDotProductAttention(dropout_rate=0.0)
    
    # Verify parameter count is 0
    num_params = count_parameters(attention_module)
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Scaled Dot Product Attention")
    logger.info(f"Total Trainable Params: {num_params} (Expected 0)")
    
    # 3. Create Dummy Tensors (representing outputs from Q, K, V linear projections)
    batch_size = 2
    num_heads = 8
    seq_len = 10
    d_k = 64 # Usually d_model / num_heads = 512 / 8 = 64
    
    # Tensors shaped for Multi-Head Attention: (batch_size, num_heads, seq_len, d_k)
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # 4. Generate Causal Mask
    # Shape: (seq_len, seq_len). It will broadcast perfectly across batch and heads.
    causal_mask = AttentionMasks.create_causal_mask(seq_len)
    
    # 5. Forward Pass
    logger.info("Running forward pass with causal mask...")
    output, p_attn = attention_module(query, key, value, mask=causal_mask)
    
    logger.info("--- TENSOR SHAPES ---")
    logger.info(f"Query/Key/Value : {query.shape}")
    logger.info(f"Causal Mask     : {causal_mask.shape}")
    logger.info(f"Attention Probs : {p_attn.shape}")
    logger.info(f"Output Matrix   : {output.shape}")
    
    # 6. Verify Mask Application visually in logs
    # Look at the first head of the first batch
    head_0_attn = p_attn[0, 0]
    logger.info("--- ATTENTION PROBABILITIES (Head 0, first 4 rows) ---")
    for i in range(4):
        # We round to 4 decimal places for readability
        row_probs = [round(val.item(), 4) for val in head_0_attn[i]]
        logger.info(f"Row {i} (Token {i} looking at past): {row_probs}")
        
    # Note how Row 0 only has probability mass on column 0.
    # Row 1 has mass on columns 0 and 1.
    # This proves the causal mask is working perfectly.
    
    # 7. Visualization
    visualizer = RepresentationVisualizer(config)
    visualizer.plot_attention_weights(p_attn, "attention_weights_heatmap.png")
    
    elapsed = time.time() - t_start
    logger.info(f"Attention pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
