import unittest
import torch
from src.llm.transformer.config import TransformerConfig
from src.llm.transformer.multi_head_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    
    def setUp(self):
        self.config = TransformerConfig(
            d_model=64, 
            num_heads=4, 
            dropout_rate=0.0
        )
        self.batch_size = 2
        self.seq_len = 10
        self.mha = MultiHeadAttention(self.config)
        
        self.query = torch.randn(self.batch_size, self.seq_len, self.config.d_model)
        self.key = torch.randn(self.batch_size, self.seq_len, self.config.d_model)
        self.value = torch.randn(self.batch_size, self.seq_len, self.config.d_model)
        
    def test_forward_pass_no_mask(self):
        output, p_attn = self.mha(self.query, self.key, self.value)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.d_model))
        # Ensure attention probabilities returned match the head split
        self.assertEqual(p_attn.shape, (self.batch_size, self.config.num_heads, self.seq_len, self.seq_len))
        
    def test_forward_pass_with_mask(self):
        # Create a simple 3D mask (batch_size, seq_len, seq_len)
        mask = torch.tril(torch.ones(self.batch_size, self.seq_len, self.seq_len, dtype=torch.bool))
        
        output, p_attn = self.mha(self.query, self.key, self.value, mask=mask)
        
        # Verify mask broadcasted properly
        self.assertEqual(p_attn.shape, (self.batch_size, self.config.num_heads, self.seq_len, self.seq_len))
        
        # Verify mask enforced causal properties across all heads
        for b in range(self.batch_size):
            for h in range(self.config.num_heads):
                for i in range(self.seq_len):
                    for j in range(i + 1, self.seq_len): # Above diagonal
                        self.assertAlmostEqual(p_attn[b, h, i, j].item(), 0.0, places=6)
                        
    def test_gradient_propagation(self):
        self.query.requires_grad_(True)
        self.key.requires_grad_(True)
        self.value.requires_grad_(True)
        
        output, _ = self.mha(self.query, self.key, self.value)
        
        loss = output.sum()
        loss.backward()
        
        # Ensure gradients flowed backward through the entire pipeline
        self.assertIsNotNone(self.query.grad)
        self.assertIsNotNone(self.mha.projections.W_q.weight.grad)
        self.assertIsNotNone(self.mha.projections.W_o.weight.grad)
        
    def test_parameter_count(self):
        # 4 layers (Q, K, V, O), each d_model x d_model, without bias
        expected_params = 4 * (self.config.d_model * self.config.d_model)
        
        actual_params = sum(p.numel() for p in self.mha.parameters())
        self.assertEqual(actual_params, expected_params)

if __name__ == '__main__':
    unittest.main()
