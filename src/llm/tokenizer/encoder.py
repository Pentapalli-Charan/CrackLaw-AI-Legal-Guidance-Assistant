import re
from typing import List, Dict, Tuple
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.vocabulary import Vocabulary
from src.llm.tokenizer.bpe import BPE

class Encoder:
    """Converts raw text into sequences of Token IDs."""
    
    def __init__(self, config: TokenizerConfig, vocab: Vocabulary, merges: Dict[Tuple[str, str], int]):
        self.config = config
        self.vocab = vocab
        self.merges = merges
        self.pattern = re.compile(self.config.pre_tokenize_pattern)
        
    def encode(self, text: str) -> List[int]:
        """Encodes text into a list of token IDs."""
        if not text:
            return []
            
        token_ids = []
        # 1. Pre-tokenize (split into words/punctuation blocks)
        # We use re.findall to extract the blocks according to the pattern
        words = self.pattern.findall(text)
        
        # 2. For each word, apply BPE merges and convert to IDs
        for word in words:
            # Get the subword tokens for this word
            bpe_tokens = BPE.encode_word(word, self.merges)
            
            # Map subwords to IDs, handling UNK
            for token in bpe_tokens:
                token_ids.append(self.vocab.get_id(token))
                
        return token_ids
