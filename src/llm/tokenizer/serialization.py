import os
import json
from typing import Dict, Tuple
from src.llm.tokenizer.config import TokenizerConfig

class Serializer:
    """Handles saving and loading the tokenizer vocabulary and merge rules."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        
    def save(self, vocab_mapping: Dict[str, int], merges: Dict[Tuple[str, str], int]) -> None:
        """Saves vocab.json, merges.txt, and tokenizer_config.json"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 1. vocab.json
        vocab_path = os.path.join(self.config.output_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_mapping, f, ensure_ascii=False, indent=2)
            
        # 2. merges.txt
        merges_path = os.path.join(self.config.output_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            for pair, rank in sorted(merges.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")
                
        # 3. tokenizer_config.json
        config_path = os.path.join(self.config.output_dir, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.config.vocab_size,
                "pad_token": self.config.pad_token,
                "unk_token": self.config.unk_token,
                "bos_token": self.config.bos_token,
                "eos_token": self.config.eos_token,
                "mask_token": self.config.mask_token,
                "cls_token": self.config.cls_token,
                "sep_token": self.config.sep_token,
                "pre_tokenize_pattern": self.config.pre_tokenize_pattern
            }, f, ensure_ascii=False, indent=2)
            
    def load(self) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int]]:
        """Loads vocab.json and merges.txt"""
        vocab_path = os.path.join(self.config.output_dir, "vocab.json")
        merges_path = os.path.join(self.config.output_dir, "merges.txt")
        
        if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
            raise FileNotFoundError(f"Missing vocab.json or merges.txt in {self.config.output_dir}")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_mapping = json.load(f)
            
        merges = {}
        with open(merges_path, "r", encoding="utf-8") as f:
            rank = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges[(parts[0], parts[1])] = rank
                    rank += 1
                    
        return vocab_mapping, merges
