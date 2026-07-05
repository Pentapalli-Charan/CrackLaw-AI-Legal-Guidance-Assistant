import torch
from typing import List, Dict, Any
from torch.nn.utils.rnn import pad_sequence
from src.llm.dataset.masking import MaskGenerator

class DataCollator:
    """
    Dynamically pads batches to the maximum sequence length in the batch.
    Also computes the padding attention masks for the batch.
    """
    
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of dataset dictionaries into a single batched dictionary.
        """
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences to max length in this specific batch
        # batch_first=True makes shape (batch_size, seq_len)
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        
        # We typically pad labels with a special value like -100 so CrossEntropyLoss ignores them.
        # Using -100 is PyTorch standard for ignore_index.
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Generate padding attention mask
        attention_mask = MaskGenerator.create_padding_mask(padded_input_ids, self.pad_token_id)
        
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels
        }
