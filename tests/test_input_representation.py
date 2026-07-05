import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.embeddings import TokenEmbedding
from src.llm.transformer.positional_encoding import SinusoidalPositionalEncoding
from src.llm.transformer.input_representation import InputRepresentation

class TestInputRepresentation(unittest.TestCase):
    
    def setUp(self):
        self.config = TransformerConfig(
            vocab_size=100, 
            d_model=16, 
            max_seq_len=50, 
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.seq_len = 10
        self.input_ids = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        
    def test_token_embedding(self):
        embedder = TokenEmbedding(self.config)
        output = embedder(self.input_ids)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        self.assertTrue(output.requires_grad)
        
        # Test padding initialization
        pad_vec = embedder.embedding.weight[0]
        self.assertTrue(torch.all(pad_vec == 0))
        
    def test_positional_encoding(self):
        pe_module = SinusoidalPositionalEncoding(self.config)
        
        # Dummy embeddings
        dummy_embeddings = torch.zeros(self.batch_size, self.seq_len, self.config.d_model)
        output = pe_module(dummy_embeddings)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        
        # Ensure PE buffer has no gradients
        self.assertFalse(pe_module.pe.requires_grad)
        
        # Check specific properties of sinusoidal PE
        # Position 0, Dimension 0 (sine) -> sin(0) = 0
        self.assertAlmostEqual(output[0, 0, 0].item(), 0.0, places=5)
        # Position 0, Dimension 1 (cosine) -> cos(0) = 1
        self.assertAlmostEqual(output[0, 0, 1].item(), 1.0, places=5)
        
    def test_input_representation(self):
        module = InputRepresentation(self.config)
        output = module(self.input_ids)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        self.assertTrue(output.requires_grad)
        
        # Test gradient propagation
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(module.token_embedding.embedding.weight.grad)
        
    def test_device_movement(self):
        # We only test CPU for absolute CI safety, but this verifies standard PyTorch module mapping
        module = InputRepresentation(self.config)
        module.to('cpu')
        
        self.assertEqual(module.positional_encoding.pe.device.type, 'cpu')
        self.assertEqual(module.token_embedding.embedding.weight.device.type, 'cpu')

if __name__ == '__main__':
    unittest.main()
