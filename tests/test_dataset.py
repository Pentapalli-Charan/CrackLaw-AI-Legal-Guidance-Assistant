import unittest
import torch
import tempfile
import json
import os
from src.llm.dataset.config import DatasetConfig
from src.llm.dataset.sequence_builder import SequenceBuilder
from src.llm.dataset.masking import MaskGenerator
from src.llm.dataset.collator import DataCollator
from src.llm.dataset.dataset import CrackLawDataset
from src.llm.dataset.dataloader import create_dataloaders
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

class MockTokenizer:
    """A mock tokenizer to bypass training for dataset testing."""
    def __init__(self):
        class MockSpecialTokens:
            def get_id(self, tok):
                return {"<BOS>": 1, "<EOS>": 2, "<PAD>": 0}.get(tok, 0)
        class MockConfig:
            bos_token = "<BOS>"
            eos_token = "<EOS>"
            pad_token = "<PAD>"
        self.special_tokens = MockSpecialTokens()
        self.config = MockConfig()
        
    def encode(self, text):
        return [10, 11, 12] # Mock encoding

class TestDatasetModule(unittest.TestCase):
    
    def setUp(self):
        self.config = DatasetConfig(max_sequence_length=5, batch_size=2)
        self.mock_tokenizer = MockTokenizer()
        
    def test_sequence_builder_truncation(self):
        builder = SequenceBuilder(self.config, self.mock_tokenizer)
        # Should truncate to max_len - 1 = 4, then add BOS/EOS
        input_ids, labels = builder.build_sequence([10, 11, 12, 13, 14, 15])
        
        # Max length is 5.
        self.assertEqual(len(input_ids), 5)
        self.assertEqual(len(labels), 5)
        self.assertEqual(input_ids[0], 1) # BOS
        self.assertEqual(labels[-1], 2) # EOS
        
    def test_mask_generation(self):
        # seq_len = 3
        causal_mask = MaskGenerator.create_causal_mask(3)
        self.assertEqual(causal_mask.shape, (3, 3))
        self.assertTrue(causal_mask[0, 0])
        self.assertFalse(causal_mask[0, 1])
        
        # padding mask
        input_ids = torch.tensor([[1, 10, 0], [1, 10, 12]])
        pad_mask = MaskGenerator.create_padding_mask(input_ids, pad_token_id=0)
        self.assertEqual(pad_mask.shape, (2, 3))
        self.assertFalse(pad_mask[0, 2].item()) # padding is False
        self.assertTrue(pad_mask[1, 2].item())
        
    def test_collator_dynamic_padding(self):
        collator = DataCollator(pad_token_id=0)
        batch = [
            {"input_ids": torch.tensor([1, 10, 2]), "labels": torch.tensor([10, 2, 0])},
            {"input_ids": torch.tensor([1, 10, 11, 2]), "labels": torch.tensor([10, 11, 2, 0])}
        ]
        
        collated = collator(batch)
        self.assertEqual(collated["input_ids"].shape, (2, 4))
        self.assertEqual(collated["labels"].shape, (2, 4))
        # Check label padding uses -100
        self.assertEqual(collated["labels"][0, 3].item(), -100)
        
    def test_dataset_and_dataloader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w") as f:
                for _ in range(10): # 10 records
                    f.write(json.dumps({"text": "dummy text that is long enough"}) + "\n")
                    
            dataset = CrackLawDataset(jsonl_path, self.config, self.mock_tokenizer)
            self.assertEqual(len(dataset), 10)
            
            # Using custom split
            config = DatasetConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, batch_size=2)
            train_loader, val_loader, test_loader = create_dataloaders(dataset, config, 0)
            
            self.assertEqual(len(train_loader.dataset), 6)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(len(test_loader.dataset), 2)
            
            # Test iteration
            for batch in train_loader:
                self.assertIn("input_ids", batch)
                self.assertIn("attention_mask", batch)
                self.assertIn("labels", batch)
                break

if __name__ == '__main__':
    unittest.main()
