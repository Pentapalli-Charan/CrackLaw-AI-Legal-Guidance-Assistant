import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.feed_forward import PositionwiseFeedForward
from src.llm.transformer.activation import ActivationFactory

class TestFeedForward(unittest.TestCase):
    
    def setUp(self):
        self.config = TransformerConfig(
            d_model=64, 
            d_ff=128, 
            dropout_rate=0.0,
            activation_function="relu"
        )
        self.batch_size = 2
        self.seq_len = 10
        self.ffn = PositionwiseFeedForward(self.config)
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.config.d_model)
        
    def test_forward_pass_dimensions(self):
        output = self.ffn(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        
    def test_relu_activation(self):
        # We manually pass the tensor through w1 and check relu works correctly
        x_expanded = self.ffn.w_1(self.input_tensor)
        # All values below 0 should become 0
        x_act = self.ffn.activation(x_expanded)
        self.assertTrue(torch.all(x_act >= 0))
        
    def test_activation_factory(self):
        act_relu = ActivationFactory.get_activation("relu")
        self.assertIsInstance(act_relu, torch.nn.ReLU)
        
        act_gelu = ActivationFactory.get_activation("gelu")
        self.assertIsInstance(act_gelu, torch.nn.GELU)
        
        act_silu = ActivationFactory.get_activation("silu")
        self.assertIsInstance(act_silu, torch.nn.SiLU)
        
        with self.assertRaises(ValueError):
            ActivationFactory.get_activation("unknown")
            
    def test_gradient_propagation(self):
        self.input_tensor.requires_grad_(True)
        output = self.ffn(self.input_tensor)
        
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(self.input_tensor.grad)
        self.assertIsNotNone(self.ffn.w_1.weight.grad)
        self.assertIsNotNone(self.ffn.w_2.weight.grad)
        
    def test_parameter_count(self):
        # Linear 1: d_model x d_ff + d_ff
        params_w1 = self.config.d_model * self.config.d_ff + self.config.d_ff
        # Linear 2: d_ff x d_model + d_model
        params_w2 = self.config.d_ff * self.config.d_model + self.config.d_model
        
        expected_params = params_w1 + params_w2
        actual_params = sum(p.numel() for p in self.ffn.parameters())
        
        self.assertEqual(actual_params, expected_params)

if __name__ == '__main__':
    unittest.main()
