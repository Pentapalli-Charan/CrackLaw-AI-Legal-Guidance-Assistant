import unittest
import torch
from src.llm.transformer.attention import ScaledDotProductAttention
from src.llm.transformer.masks import AttentionMasks

class TestScaledDotProductAttention(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 5
        self.seq_len_k = 5
        self.d_k = 16
        self.d_v = 16
        
        self.query = torch.randn(self.batch_size, self.seq_len_q, self.d_k)
        self.key = torch.randn(self.batch_size, self.seq_len_k, self.d_k)
        self.value = torch.randn(self.batch_size, self.seq_len_k, self.d_v)
        
        # Dropout set to 0.0 for deterministic testing
        self.attention = ScaledDotProductAttention(dropout_rate=0.0)
        
    def test_forward_pass_no_mask(self):
        output, p_attn = self.attention(self.query, self.key, self.value)
        
        # Check shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len_q, self.d_v))
        self.assertEqual(p_attn.shape, (self.batch_size, self.seq_len_q, self.seq_len_k))
        
        # Check probabilities sum to 1
        sum_probs = p_attn.sum(dim=-1)
        # Verify that all sums are very close to 1.0
        self.assertTrue(torch.allclose(sum_probs, torch.ones_like(sum_probs)))
        
    def test_forward_pass_with_mask(self):
        # Create a causal mask for a square matrix (seq_len_q == seq_len_k)
        mask = AttentionMasks.create_causal_mask(self.seq_len_q)
        # Expand mask to batch shape for testing 
        # (attention module broadcasts it, but we can provide explicit shape too)
        mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        output, p_attn = self.attention(self.query, self.key, self.value, mask=mask)
        
        # Ensure that masked out positions are strictly 0.0
        # Causal mask should have 0s above the diagonal
        for b in range(self.batch_size):
            for i in range(self.seq_len_q):
                for j in range(i + 1, self.seq_len_k): # Above diagonal
                    self.assertAlmostEqual(p_attn[b, i, j].item(), 0.0, places=6)
                    
        # Check probabilities still sum to 1
        sum_probs = p_attn.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_probs, torch.ones_like(sum_probs)))
        
    def test_gradient_propagation(self):
        self.query.requires_grad_(True)
        self.key.requires_grad_(True)
        self.value.requires_grad_(True)
        
        output, _ = self.attention(self.query, self.key, self.value)
        
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(self.query.grad)
        self.assertIsNotNone(self.key.grad)
        self.assertIsNotNone(self.value.grad)
        
    def test_zero_parameters(self):
        params = list(self.attention.parameters())
        self.assertEqual(len(params), 0)

if __name__ == '__main__':
    unittest.main()
