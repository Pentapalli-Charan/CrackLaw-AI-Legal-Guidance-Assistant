import os
import torch
import logging
import time
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.multi_head_attention import MultiHeadAttention
from src.llm.transformer.masks import AttentionMasks
from src.llm.transformer.visualization import RepresentationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.RunMultiHeadAttention")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    t_start = time.time()
    logger.info("Initializing Multi-Head Attention Module...")
    
    # 1. Setup Configuration
    config = TransformerConfig(
        vocab_size=5000,
        d_model=512,
        num_heads=8,
        dropout_rate=0.0  # Set to 0 for deterministic log output
    )
    
    # 2. Instantiate Architecture
    mha_module = MultiHeadAttention(config)
    
    # Verify parameter count
    num_params = count_parameters(mha_module)
    expected_params = 4 * (config.d_model * config.d_model) # W_q, W_k, W_v, W_o
    
    logger.info("--- ARCHITECTURE REPORT ---")
    logger.info(f"Module Type          : Multi-Head Attention")
    logger.info(f"Dimensions (d_model) : {config.d_model}")
    logger.info(f"Number of Heads      : {config.num_heads}")
    logger.info(f"Head Dim (d_k/d_v)   : {config.d_k}")
    logger.info(f"Total Trainable Params: {num_params:,} (Expected: {expected_params:,})")
    
    # 3. Create Dummy Tensors (representing outputs from Input Representation stage)
    batch_size = 2
    seq_len = 10
    
    # Tensors shaped for Multi-Head Attention input: (batch_size, seq_len, d_model)
    # We use self-attention here, so Query = Key = Value
    hidden_states = torch.randn(batch_size, seq_len, config.d_model)
    
    # 4. Generate Causal Mask
    # Shape: (seq_len, seq_len)
    causal_mask = AttentionMasks.create_causal_mask(seq_len)
    
    # 5. Forward Pass
    logger.info("Running forward pass with causal mask...")
    output, p_attn = mha_module(query=hidden_states, key=hidden_states, value=hidden_states, mask=causal_mask)
    
    logger.info("--- TENSOR SHAPES ---")
    logger.info(f"Input Vectors   : {hidden_states.shape} (batch_size, seq_len, d_model)")
    logger.info(f"Causal Mask     : {causal_mask.shape} (seq_len, seq_len)")
    logger.info(f"Attention Probs : {p_attn.shape} (batch_size, num_heads, seq_len, seq_len)")
    logger.info(f"Final Output    : {output.shape} (batch_size, seq_len, d_model)")
    
    # Calculate Memory
    mem_mb = (output.element_size() * output.nelement()) / (1024 * 1024)
    logger.info(f"Output Matrix Memory Footprint: {mem_mb:.4f} MB")
    
    # 6. Visualization
    # Plot attention for the very first head
    visualizer = RepresentationVisualizer(config)
    visualizer.plot_attention_weights(p_attn, "multi_head_attention_heatmap.png")
    
    elapsed = time.time() - t_start
    logger.info(f"Multi-Head Attention pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
