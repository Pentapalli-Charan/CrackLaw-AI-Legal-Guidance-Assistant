import torch
from src.llm.transformer.model import CrackLawTransformer

class TransformerGenerator:
    """
    Inference Engine interface for the CrackLaw Transformer.
    This class wraps the auto-regressive decoding loops needed to generate text.
    Currently implements Greedy Search, and provides structural hooks for advanced sampling.
    """
    def __init__(self, model: CrackLawTransformer):
        self.model = model
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def greedy_search(
        self, 
        src_input_ids: torch.Tensor, 
        src_padding_mask: torch.Tensor, 
        max_new_tokens: int, 
        start_token_id: int, 
        end_token_id: int
    ) -> torch.Tensor:
        """
        Auto-regressively generates tokens by always picking the highest probability next token.
        
        Args:
            src_input_ids: (1, src_seq_len)
            src_padding_mask: (1, 1, 1, src_seq_len)
            max_new_tokens: Maximum length to generate
            start_token_id: The <SOS> token ID to prime the decoder
            end_token_id: The <EOS> token ID to stop generation early
            
        Returns:
            Generated token IDs.
        """
        # Ensure model is in inference mode (disables dropout, etc.)
        self.model.eval()
        
        # 1. Pre-compute Encoder states (Context caching)
        # We only need to run the heavy Encoder ONCE
        encoder_hidden_states = self.model.encode(src_input_ids, src_padding_mask)
        
        # 2. Initialize the Decoder sequence with the start token
        batch_size = src_input_ids.size(0)
        generated_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=self.device)
        
        # 3. Auto-regressive loop
        for _ in range(max_new_tokens):
            # Run the Decoder using the cached Encoder context
            decoder_hidden_states = self.model.decode(
                tgt_input_ids=generated_ids, 
                encoder_hidden_states=encoder_hidden_states, 
                src_padding_mask=src_padding_mask
            )
            
            # Project to vocabulary (we only care about the very last token generated)
            next_token_hidden = decoder_hidden_states[:, -1, :]  # Shape: (batch_size, d_model)
            logits = self.model.generate_logits(next_token_hidden) # Shape: (batch_size, vocab_size)
            
            # Greedy pick (argmax)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Stop early if <EOS> is generated (assuming batch_size=1 for simple generation)
            if next_token_id.item() == end_token_id:
                break
                
        return generated_ids

    # --- Structural Hooks for Future Advanced Decoding Implementations ---
    
    @torch.no_grad()
    def beam_search(self, *args, **kwargs):
        raise NotImplementedError("Beam search decoding will be implemented in a future iteration.")
        
    @torch.no_grad()
    def top_k_sampling(self, *args, **kwargs):
        raise NotImplementedError("Top-K sampling will be implemented in a future iteration.")
        
    @torch.no_grad()
    def top_p_sampling(self, *args, **kwargs):
        raise NotImplementedError("Nucleus (Top-p) sampling will be implemented in a future iteration.")
