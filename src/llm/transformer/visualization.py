import os
import torch
import logging

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.llm.transformer.config import TransformerConfig

logger = logging.getLogger("CrackLaw.LLM.Visualization")

class RepresentationVisualizer:
    """Utilities to visualize embedding vectors and positional encodings."""
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        
    def plot_positional_encoding(self, pe_tensor: torch.Tensor, filename: str = "positional_encoding.png"):
        """
        Plots a heatmap of the positional encoding matrix.
        pe_tensor should be of shape (max_seq_len, d_model)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib is not installed. Skipping visualization.")
            return
            
        pe_np = pe_tensor.squeeze(0).cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        # Plotting the sequence length on Y-axis and dimension on X-axis
        # Often it's clearer to plot a subset, e.g., first 100 positions and first 100 dimensions
        subset_len = min(100, pe_np.shape[0])
        subset_dim = min(100, pe_np.shape[1])
        
        cax = plt.matshow(pe_np[:subset_len, :subset_dim], cmap='viridis', fignum=1)
        plt.colorbar(cax)
        
        plt.title('Sinusoidal Positional Encoding (First 100 pos x 100 dims)')
        plt.ylabel('Sequence Position')
        plt.xlabel('Embedding Dimension')
        
        filepath = os.path.join(self.config.visualization_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Positional Encoding heatmap to {filepath}")
        
    def generate_tensor_report(self, input_ids: torch.Tensor, output_tensor: torch.Tensor):
        """Logs standard information about the processed tensors."""
        logger.info(f"Input IDs Shape: {input_ids.shape} (batch_size, seq_len)")
        logger.info(f"Output Representation Shape: {output_tensor.shape} (batch_size, seq_len, d_model)")
        
        mem_mb = (output_tensor.element_size() * output_tensor.nelement()) / (1024 * 1024)
        logger.info(f"Output Tensor Memory Footprint: {mem_mb:.4f} MB")

    def plot_attention_weights(self, p_attn: torch.Tensor, filename: str = "attention_weights.png"):
        """
        Plots a heatmap of the attention probabilities.
        p_attn should be of shape (batch_size, seq_len_q, seq_len_k) or (batch_size, num_heads, seq_len_q, seq_len_k).
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib is not installed. Skipping attention visualization.")
            return
            
        # If it has 4 dimensions, just take the first head of the first batch
        if p_attn.dim() == 4:
            attn_np = p_attn[0, 0].detach().cpu().numpy()
        else:
            attn_np = p_attn[0].detach().cpu().numpy()
            
        plt.figure(figsize=(8, 8))
        
        # We plot the first 50x50 tokens to keep it readable
        subset_len = min(50, attn_np.shape[0])
        
        cax = plt.matshow(attn_np[:subset_len, :subset_len], cmap='viridis', fignum=1)
        plt.colorbar(cax)
        
        plt.title('Attention Probabilities (First 50x50 tokens)')
        plt.ylabel('Query Position')
        plt.xlabel('Key Position')
        
        filepath = os.path.join(self.config.visualization_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Attention Weights heatmap to {filepath}")

    def plot_activation_distribution(self, pre_act: torch.Tensor, post_act: torch.Tensor, filename: str = "activation_distribution.png"):
        """
        Plots the histogram distribution of values before and after the activation function.
        Helps in diagnosing dead neurons (e.g. in ReLU) or saturated representations.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib is not installed. Skipping activation visualization.")
            return
            
        pre_np = pre_act.detach().cpu().numpy().flatten()
        post_np = post_act.detach().cpu().numpy().flatten()
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pre-activation Histogram
        axs[0].hist(pre_np, bins=50, color='blue', alpha=0.7)
        axs[0].set_title(f'Pre-Activation Distribution\nMean: {pre_np.mean():.4f}, Std: {pre_np.std():.4f}')
        axs[0].set_xlabel('Value')
        axs[0].set_ylabel('Frequency')
        
        # Post-activation Histogram
        axs[1].hist(post_np, bins=50, color='orange', alpha=0.7)
        axs[1].set_title(f'Post-Activation ({self.config.activation_function.upper()}) Distribution\nMean: {post_np.mean():.4f}, Std: {post_np.std():.4f}')
        axs[1].set_xlabel('Value')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        filepath = os.path.join(self.config.visualization_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Activation Distribution to {filepath}")

    def plot_normalization_statistics(self, pre_norm: torch.Tensor, post_norm: torch.Tensor, filename: str = "normalization_statistics.png"):
        """
        Plots histograms proving that LayerNorm correctly shifts mean to 0 and variance to 1.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib is not installed. Skipping normalization visualization.")
            return
            
        # We'll take a single token's feature vector across the d_model dimension
        # Shape: (d_model,)
        pre_np = pre_norm[0, 0].detach().cpu().numpy()
        post_np = post_norm[0, 0].detach().cpu().numpy()
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pre-Norm Histogram
        axs[0].hist(pre_np, bins=30, color='red', alpha=0.7)
        axs[0].set_title(f'Pre-Norm (Single Token)\nMean: {pre_np.mean():.4f}, Std: {pre_np.std():.4f}')
        axs[0].set_xlabel('Feature Value')
        axs[0].set_ylabel('Frequency')
        
        # Post-Norm Histogram
        axs[1].hist(post_np, bins=30, color='green', alpha=0.7)
        axs[1].set_title(f'Post-Norm (Single Token)\nMean: {post_np.mean():.4f}, Std: {post_np.std():.4f}')
        axs[1].set_xlabel('Feature Value')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        filepath = os.path.join(self.config.visualization_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Normalization Statistics to {filepath}")

    def generate_encoder_diagram(self, filename: str = "encoder_architecture.txt"):
        """
        Generates a textual diagram of the initialized Encoder stack to log file.
        """
        diagram = [
            "====================================================",
            "             TRANSFORMER ENCODER STACK              ",
            "====================================================",
            f"Input IDs -> Shape: (batch_size, seq_len)",
            "  ↓",
            f"Input Representation (Vocab: {self.config.vocab_size}, Dim: {self.config.d_model})",
            "  ↓"
        ]
        
        for i in range(self.config.num_encoder_layers):
            diagram.extend([
                f"--- Encoder Block {i+1} ---",
                "    |-> Multi-Head Attention",
                "    |   -> Residual + LayerNorm",
                "    |-> Feed Forward Network",
                "    |   -> Residual + LayerNorm",
                "  ↓"
            ])
            
        diagram.extend([
            "Final Layer Normalization",
            "  ↓",
            f"Encoded Hidden States -> Shape: (batch_size, seq_len, {self.config.d_model})",
            "===================================================="
        ])
        
        filepath = os.path.join(self.config.visualization_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(diagram))
            
        for line in diagram:
            logger.info(line)
        
        logger.info(f"Saved Encoder Diagram to {filepath}")

    def generate_decoder_diagram(self, filename: str = "decoder_architecture.txt"):
        """
        Generates a textual diagram of the initialized Decoder stack to log file.
        """
        diagram = [
            "====================================================",
            "             TRANSFORMER DECODER STACK              ",
            "====================================================",
            f"Target IDs -> Shape: (batch_size, tgt_seq_len)",
            "  ↓",
            f"Input Representation (Vocab: {self.config.vocab_size}, Dim: {self.config.d_model})",
            "  ↓"
        ]
        
        for i in range(self.config.num_decoder_layers):
            diagram.extend([
                f"--- Decoder Block {i+1} ---",
                "    |-> Masked Multi-Head Self-Attention",
                "    |   -> Residual + LayerNorm",
                "    |-> Cross-Attention (with Encoder States)",
                "    |   -> Residual + LayerNorm",
                "    |-> Feed Forward Network",
                "    |   -> Residual + LayerNorm",
                "  ↓"
            ])
            
        diagram.extend([
            "Final Layer Normalization",
            "  ↓",
            f"Decoded Hidden States -> Shape: (batch_size, tgt_seq_len, {self.config.d_model})",
            "===================================================="
        ])
        
        filepath = os.path.join(self.config.visualization_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(diagram))
            
        for line in diagram:
            logger.info(line)
        
        logger.info(f"Saved Decoder Diagram to {filepath}")

    def generate_full_model_diagram(self, filename: str = "full_model_architecture.txt"):
        """
        Generates a textual diagram of the initialized full Transformer model to log file.
        """
        diagram = [
            "====================================================",
            "             CRACKLAW TRANSFORMER                   ",
            "====================================================",
            "                   [ ENCODER ]                      ",
            f"Source Tokens (Vocab: {self.config.vocab_size})",
            "  ↓",
            f"Encoder Stack ({self.config.num_encoder_layers} Layers)",
            "  ↓",
            f"Encoder Context (d_model: {self.config.d_model})",
            "  ↓",
            "====================================================",
            "                   [ DECODER ]                      ",
            f"Target Tokens (Vocab: {self.config.vocab_size})",
            "  ↓",
            f"Decoder Stack ({self.config.num_decoder_layers} Layers) <---- [Cross Attention to Encoder Context]",
            "  ↓",
            "====================================================",
            "                   [ LM HEAD ]                      ",
            f"Linear Projection ({self.config.d_model} -> {self.config.vocab_size})",
            f"Weight Tying: {'ENABLED' if self.config.tie_word_embeddings else 'DISABLED'}",
            "  ↓",
            "Vocabulary Logits",
            "===================================================="
        ]
        
        filepath = os.path.join(self.config.visualization_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(diagram))
            
        for line in diagram:
            logger.info(line)
        
        logger.info(f"Saved Full Model Diagram to {filepath}")


