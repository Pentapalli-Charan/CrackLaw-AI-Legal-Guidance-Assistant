import re
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
from src.llm.tokenizer.config import TokenizerConfig
from src.llm.tokenizer.vocabulary import Vocabulary
from src.llm.tokenizer.bpe import BPE

logger = logging.getLogger("CrackLaw.LLM.TokenizerTrainer")

class Trainer:
    """Trains the BPE Tokenizer on a corpus of text."""
    
    def __init__(self, config: TokenizerConfig, vocab: Vocabulary):
        self.config = config
        self.vocab = vocab
        self.merges: Dict[Tuple[str, str], int] = {}
        self.pattern = re.compile(self.config.pre_tokenize_pattern)
        
    def train(self, texts: List[str]) -> Dict[Tuple[str, str], int]:
        """Runs the BPE training loop on the provided texts."""
        
        logger.info(f"Starting BPE training for {len(texts)} documents...")
        logger.info(f"Target vocabulary size: {self.config.vocab_size}")
        
        # 1. Pre-tokenize and count word frequencies
        word_counts = defaultdict(int)
        
        for text in texts:
            words = self.pattern.findall(text)
            for word in words:
                word_counts[word] += 1
                
        # 2. Initial splits: break each word into a tuple of characters
        # e.g. "hello" -> ('h', 'e', 'l', 'l', 'o')
        splits = {tuple(word): count for word, count in word_counts.items()}
        
        # 3. Add base characters to vocabulary
        base_chars = set()
        for word in splits.keys():
            for char in word:
                base_chars.add(char)
                
        for char in sorted(list(base_chars)):
            self.vocab.add_token(char)
            
        logger.info(f"Base alphabet size: {len(base_chars)}")
        
        # 4. BPE Merge Loop
        num_merges = self.config.vocab_size - len(self.vocab)
        if num_merges <= 0:
            logger.warning("Target vocab size is smaller than base alphabet + special tokens.")
            return self.merges
            
        logger.info(f"Performing up to {num_merges} merges...")
        
        # Progress bar for training
        pbar = tqdm(total=num_merges, desc="BPE Merges")
        
        for i in range(num_merges):
            # Compute stats for adjacent pairs
            stats = BPE.get_stats(splits)
            if not stats:
                logger.info("No more pairs to merge.")
                break
                
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            
            # Merge the best pair
            splits = BPE.merge_vocab(best_pair, splits)
            
            # Save the merge rule (rank = i)
            self.merges[best_pair] = i
            
            # Add the new merged token to vocabulary
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add_token(new_token)
            
            pbar.update(1)
            
        pbar.close()
        logger.info(f"BPE training complete. Final vocabulary size: {len(self.vocab)}")
        return self.merges
