import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.decoder import Decoder
from src.llm.transformer.masks import AttentionMasks

class TestDecoder(unittest.TestCase):
    
    def setUp(self):
        # We use a small config for fast testing
        self.config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_decoder_layers=2, # Stack of 2
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.src_seq_len = 12
        self.tgt_seq_len = 8
        
        self.tgt_input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.tgt_seq_len))
        self.encoder_hidden_states = torch.randn(self.batch_size, self.src_seq_len, self.config.d_model)
        
        self.decoder = Decoder(self.config)
        
    def test_forward_pass_dimensions(self):
        # Basic causal mask
        causal_mask = AttentionMasks.create_causal_mask(self.tgt_seq_len)
        
        output = self.decoder(
            tgt_input_ids=self.tgt_input_ids,
            encoder_hidden_states=self.encoder_hidden_states,
            tgt_mask=causal_mask
        )
        self.assertEqual(output.shape, (self.batch_size, self.tgt_seq_len, self.config.d_model))
        
    def test_stack_depth(self):
        self.assertEqual(len(self.decoder.decoder_stack.layers), self.config.num_decoder_layers)
        
    def test_gradient_propagation(self):
        # We want to make sure gradients flow backwards through the decoder AND into the encoder hidden states
        self.encoder_hidden_states.requires_grad_(True)
        
        output = self.decoder(
            tgt_input_ids=self.tgt_input_ids,
            encoder_hidden_states=self.encoder_hidden_states
        )
        loss = output.sum()
        loss.backward()
        
        # Verify gradients reached the target embedding layer
        self.assertIsNotNone(self.decoder.input_representation.token_embedding.embedding.weight.grad)
        
        # Verify gradients reached the Encoder hidden states (meaning Cross-Attention works)
        self.assertIsNotNone(self.encoder_hidden_states.grad)

if __name__ == '__main__':
    unittest.main()
