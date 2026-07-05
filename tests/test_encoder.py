import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.encoder import Encoder

class TestEncoder(unittest.TestCase):
    
    def setUp(self):
        # We use a small config for fast testing
        self.config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_encoder_layers=2, # Stack of 2
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.seq_len = 10
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        self.encoder = Encoder(self.config)
        
    def test_forward_pass_dimensions(self):
        output = self.encoder(self.input_ids)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        
    def test_stack_depth(self):
        self.assertEqual(len(self.encoder.encoder_stack.layers), self.config.num_encoder_layers)
        
    def test_gradient_propagation(self):
        output = self.encoder(self.input_ids)
        loss = output.sum()
        loss.backward()
        
        # Verify gradients reached the very first layer (Token Embedding)
        self.assertIsNotNone(self.encoder.input_representation.token_embedding.embedding.weight.grad)
        
        # Verify gradients reached the deep layers (Multi-Head Attention of the first block)
        self.assertIsNotNone(self.encoder.encoder_stack.layers[0].self_attention.projections.W_q.weight.grad)

if __name__ == '__main__':
    unittest.main()
