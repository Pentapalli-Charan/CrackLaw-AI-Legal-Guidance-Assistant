import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import logging

from src.llm.dataset.config import DatasetConfig
from src.llm.dataset.preprocessing import DataPreprocessor
from src.llm.dataset.sequence_builder import SequenceBuilder
from src.llm.tokenizer.tokenizer import CrackLawTokenizer

logger = logging.getLogger("CrackLaw.LLM.Dataset")

class CrackLawDataset(Dataset):
    """
    Native PyTorch Dataset for loading CrackLaw corpus into training tensors.
    """
    
    def __init__(self, jsonl_path: str, config: DatasetConfig, tokenizer: CrackLawTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.sequence_builder = SequenceBuilder(config, tokenizer)
        
        # We load data eagerly into memory. If the dataset gets massive,
        # this should be converted to an IterableDataset or use memmap.
        self.data: List[Tuple[List[int], List[int]]] = []
        self._load_data(jsonl_path)
        
    def _load_data(self, path: str):
        logger.info(f"Loading dataset from {path}...")
        valid_samples = 0
        dropped_samples = 0
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text", "")
                
                # Preprocess
                text = DataPreprocessor.normalize_text(text)
                if not DataPreprocessor.is_valid_sample(text):
                    dropped_samples += 1
                    continue
                    
                # Tokenize
                token_ids = self.tokenizer.encode(text)
                if not token_ids:
                    dropped_samples += 1
                    continue
                    
                # Build Sequence
                input_ids, labels = self.sequence_builder.build_sequence(token_ids)
                
                self.data.append((input_ids, labels))
                valid_samples += 1
                
        logger.info(f"Loaded {valid_samples} valid samples. Dropped {dropped_samples}.")

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.data[idx]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
