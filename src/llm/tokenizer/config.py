import os
from dataclasses import dataclass, field
from typing import List
from src.config import PROJECT_ROOT

@dataclass
class TokenizerConfig:
    """Configuration for the CrackLaw BPE Tokenizer."""
    
    # Target vocabulary size
    vocab_size: int = 32000
    
    # Pre-tokenization regex pattern (splits by space/punctuation)
    # This prevents the BPE algorithm from merging characters across words.
    pre_tokenize_pattern: str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    
    # Output directory for serialization
    output_dir: str = os.path.join(PROJECT_ROOT, "src", "llm", "tokenizer", "models")
    
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    cls_token: str = "<CLS>"
    sep_token: str = "<SEP>"

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
