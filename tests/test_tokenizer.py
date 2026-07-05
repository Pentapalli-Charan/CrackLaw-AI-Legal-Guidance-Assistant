import unittest
import os
import tempfile
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

class TestTokenizer(unittest.TestCase):
    
    def setUp(self):
        self.config = TokenizerConfig(vocab_size=30)
        self.tokenizer = CrackLawTokenizer(self.config)
        self.sample_text = ["hello world", "hello again", "world of hello"]
        
    def test_training_and_encoding(self):
        # Train on a tiny dataset
        self.tokenizer.train(self.sample_text)
        
        # Test vocabulary grew properly
        # special tokens (7) + unique characters + merged pairs
        self.assertGreater(self.tokenizer.get_vocab_size(), 7)
        
        # Test encoding
        token_ids = self.tokenizer.encode("hello")
        self.assertTrue(len(token_ids) > 0)
        
        # Test decoding
        decoded = self.tokenizer.decode(token_ids)
        self.assertEqual(decoded, "hello")
        
    def test_serialization(self):
        self.tokenizer.train(self.sample_text)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.config.output_dir = tmpdir
            self.tokenizer.save()
            
            # Check files exist
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "vocab.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "merges.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tokenizer_config.json")))
            
            # Load into a new tokenizer
            new_tokenizer = CrackLawTokenizer(TokenizerConfig(output_dir=tmpdir))
            new_tokenizer.load()
            
            self.assertEqual(self.tokenizer.get_vocab_size(), new_tokenizer.get_vocab_size())
            
            # Ensure encoding matches exactly after load
            original_encoding = self.tokenizer.encode("world")
            new_encoding = new_tokenizer.encode("world")
            self.assertEqual(original_encoding, new_encoding)
            
    def test_unknown_token_handling(self):
        self.tokenizer.train(["abcdefg"])
        # 'z' is not in vocabulary, should map to UNK
        encoded = self.tokenizer.encode("z")
        unk_id = self.tokenizer.special_tokens.get_id(self.config.unk_token)
        self.assertIn(unk_id, encoded)
        
        decoded = self.tokenizer.decode(encoded)
        # UNK token gets output literally during decoding
        self.assertIn(self.config.unk_token, decoded)

if __name__ == '__main__':
    unittest.main()
