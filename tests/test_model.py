import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.model import CrackLawTransformer
from src.llm.transformer.generation import TransformerGenerator

class TestTransformer(unittest.TestCase):
    
    def setUp(self):
        # We use a small config for fast testing
        self.config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_encoder_layers=2, 
            num_decoder_layers=2,
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.src_seq_len = 12
        self.tgt_seq_len = 8
        
        self.src_input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt_input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.tgt_seq_len))
        
        self.model = CrackLawTransformer(self.config)
        
    def test_forward_pass_dimensions(self):
        # Teacher forcing forward pass
        logits = self.model(
            src_input_ids=self.src_input_ids,
            tgt_input_ids=self.tgt_input_ids
        )
        
        # Logits should map to Vocab Size
        self.assertEqual(logits.shape, (self.batch_size, self.tgt_seq_len, self.config.vocab_size))
        
    def test_weight_tying(self):
        tied_config = TransformerConfig(vocab_size=100, d_model=32, num_heads=4, tie_word_embeddings=True)
        tied_model = CrackLawTransformer(tied_config)
        
        encoder_emb = tied_model.model.encoder.input_representation.token_embedding.embedding.weight
        decoder_emb = tied_model.model.decoder.input_representation.token_embedding.embedding.weight
        lm_head_weight = tied_model.model.lm_head.projection.weight
        
        # In PyTorch, using `id()` or checking `is` confirms they point to the exact same memory tensor
        self.assertIs(encoder_emb, decoder_emb)
        self.assertIs(decoder_emb, lm_head_weight)
        
    def test_gradient_propagation(self):
        logits = self.model(
            src_input_ids=self.src_input_ids,
            tgt_input_ids=self.tgt_input_ids
        )
        loss = logits.sum()
        loss.backward()
        
        # Verify gradients reached the LM Head
        self.assertIsNotNone(self.model.model.lm_head.projection.weight.grad)
        
        # Verify gradients reached the very beginning (Encoder Source Embeddings)
        self.assertIsNotNone(self.model.model.encoder.input_representation.token_embedding.embedding.weight.grad)

    def test_greedy_generation(self):
        generator = TransformerGenerator(self.model)
        src_mask = torch.ones(self.batch_size, 1, 1, self.src_seq_len, dtype=torch.bool)
        
        start_token = 1
        end_token = 2
        max_tokens = 5
        
        # Running greedy search with batch_size=1 as standard for loop inference
        generated_ids = generator.greedy_search(
            src_input_ids=self.src_input_ids[0:1],
            src_padding_mask=src_mask[0:1],
            max_new_tokens=max_tokens,
            start_token_id=start_token,
            end_token_id=end_token
        )
        
        # Output should be initial start_token + generated sequence (up to max_tokens)
        self.assertTrue(generated_ids.size(1) <= max_tokens + 1)
        self.assertEqual(generated_ids[0, 0].item(), start_token)

if __name__ == '__main__':
    unittest.main()
