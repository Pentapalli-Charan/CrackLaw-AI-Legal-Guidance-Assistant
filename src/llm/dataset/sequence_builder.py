from typing import List, Tuple
from src.llm.dataset.config import DatasetConfig
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

class SequenceBuilder:
    """Constructs input and target sequences for causal language modeling."""
    
    def __init__(self, config: DatasetConfig, tokenizer: CrackLawTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Cache special tokens
        self.bos_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.bos_token)
        self.eos_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.eos_token)
        self.pad_id = self.tokenizer.special_tokens.get_id(self.tokenizer.config.pad_token)
        
    def build_sequence(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Builds causal language modeling sequences.
        Inputs:  [BOS, t_1, t_2, ..., t_n-1]
        Targets: [t_1, t_2, ..., t_n-1, EOS]
        Returns: (input_ids, labels)
        """
        # We need space for at least BOS/EOS
        max_len = self.config.max_sequence_length
        max_tokens = max_len - 1 # Since we append one special token to input/target respectively
        
        # Truncate if necessary
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            
        # Build Input (BOS + tokens)
        input_ids = [self.bos_id] + token_ids
        
        # Build Target (tokens + EOS)
        labels = token_ids + [self.eos_id]
        
        # Hard padding if requested (otherwise we leave for dynamic padding in collator)
        if self.config.pad_to_max_length:
            padding_length = max_len - len(input_ids)
            input_ids.extend([self.pad_id] * padding_length)
            # Typically in PyTorch CrossEntropyLoss, padding index can be ignored.
            # Some use -100 to ignore, but we will keep pad_id and use attention masks or ignore_index.
            labels.extend([self.pad_id] * padding_length)
            
        return input_ids, labels
