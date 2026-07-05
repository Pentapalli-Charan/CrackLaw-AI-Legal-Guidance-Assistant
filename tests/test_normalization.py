import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.layer_norm import LayerNorm
from src.llm.transformer.residual import ResidualConnection
from src.llm.transformer.normalization import SublayerConnection

class TestNormalization(unittest.TestCase):
    
    def setUp(self):
        self.config = TransformerConfig(
            d_model=64, 
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.seq_len = 10
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model) * 5.0 + 10.0 # Shift mean and scale var
        
    def test_layer_norm_math(self):
        ln = LayerNorm(self.config)
        output = ln(self.input_tensor)
        
        self.assertEqual(output.shape, self.input_tensor.shape)
        
        # Mean across d_model should be extremely close to 0
        means = output.mean(dim=-1)
        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))
        
        # Variance across d_model should be extremely close to 1
        # Use unbiased=False to match our LayerNorm implementation
        variances = output.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(variances, torch.ones_like(variances), atol=1e-3))
        
    def test_residual_connection(self):
        res = ResidualConnection(self.config)
        sublayer_out = torch.ones_like(self.input_tensor)
        
        output = res(self.input_tensor, sublayer_out)
        
        self.assertEqual(output.shape, self.input_tensor.shape)
        # Without dropout, output should exactly be input + sublayer_out
        self.assertTrue(torch.all(output == self.input_tensor + sublayer_out))
        
    def test_sublayer_connection_post_norm(self):
        # Default config is norm_first=False (Post-Norm)
        sub = SublayerConnection(self.config)
        
        # Dummy sublayer that just returns ones
        def dummy_sublayer(x):
            return torch.ones_like(x)
            
        output = sub(self.input_tensor, dummy_sublayer)
        self.assertEqual(output.shape, self.input_tensor.shape)
        
        # In Post-Norm, the final operation is LayerNorm, so output mean should be 0
        means = output.mean(dim=-1)
        self.assertTrue(torch.allclose(means, torch.zeros_like(means), atol=1e-5))

    def test_sublayer_connection_pre_norm(self):
        config = TransformerConfig(d_model=64, dropout_rate=0.0, norm_first=True)
        sub = SublayerConnection(config)
        
        def dummy_sublayer(x):
            return x * 2.0
            
        output = sub(self.input_tensor, dummy_sublayer)
        self.assertEqual(output.shape, self.input_tensor.shape)
        
        # In Pre-Norm, the final operation is Residual Addition (x + Sublayer(Norm(x))).
        # Therefore, the final output mean is NOT forced to 0.
        means = output.mean(dim=-1)
        self.assertFalse(torch.allclose(means, torch.zeros_like(means), atol=1e-5))
        
    def test_gradient_propagation(self):
        ln = LayerNorm(self.config)
        self.input_tensor.requires_grad_(True)
        
        output = ln(self.input_tensor)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(self.input_tensor.grad)
        self.assertIsNotNone(ln.gamma.grad)
        self.assertIsNotNone(ln.beta.grad)

if __name__ == '__main__':
    unittest.main()
