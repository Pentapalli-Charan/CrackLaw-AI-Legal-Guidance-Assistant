from typing import Dict, List, Tuple
from collections import defaultdict

class BPE:
    """Core logic for Byte Pair Encoding (BPE) operations."""
    
    @staticmethod
    def get_stats(splits: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        Computes frequencies of all adjacent pairs of symbols.
        `splits` is a dict mapping a word (as a tuple of symbols) to its frequency.
        """
        pairs = defaultdict(int)
        for word, freq in splits.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] += freq
        return pairs

    @staticmethod
    def merge_vocab(pair: Tuple[str, str], splits: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        Merges all occurrences of the most frequent pair in the splits dictionary.
        Returns a new splits dictionary.
        """
        new_splits = {}
        first, second = pair
        
        for word, freq in splits.items():
            # If the pair doesn't exist in this word, just copy it
            if first not in word:
                new_splits[word] = freq
                continue
                
            new_word = []
            i = 0
            while i < len(word):
                # Check for the pair match
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_splits[tuple(new_word)] = freq
            
        return new_splits
        
    @staticmethod
    def encode_word(word: str, merges: Dict[Tuple[str, str], str]) -> List[str]:
        """
        Applies learned BPE merges to a single word to tokenize it.
        """
        if not word:
            return []
            
        # Initial split: individual characters
        splits = list(word)
        
        while len(splits) >= 2:
            # Find the pair of adjacent symbols in the current split
            # that was merged earliest in training (has the lowest rank/highest priority)
            
            # Get all pairs in current split
            pairs = [(splits[i], splits[i+1]) for i in range(len(splits) - 1)]
            
            # Find the pair that exists in merges. If none exist, we are done.
            # We want the pair that occurs in merges. To strictly follow priority,
            # merges should ideally be a dict mapping pair -> rank.
            # For simplicity, we just find any applicable merge. Since we apply them
            # greedily, standard BPE finds the pair with the lowest merge index.
            
            # Note: A proper implementation checks the rank of the merge. 
            # We assume `merges` is a dict of {pair: rank}, so we find the min rank.
            best_pair = None
            best_rank = float('inf')
            
            for i, pair in enumerate(pairs):
                if pair in merges:
                    rank = merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                        
            if best_pair is None:
                break # No more merges can be applied
                
            # Apply the best pair merge
            first, second = best_pair
            new_splits = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == first and splits[i+1] == second:
                    new_splits.append(first + second)
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            splits = new_splits
            
        return splits
