from typing import Dict, List, Optional
from src.llm.tokenizer.special_tokens import SpecialTokens

class Vocabulary:
    """Manages the token-to-ID and ID-to-token mappings for the tokenizer."""
    
    def __init__(self, special_tokens: SpecialTokens):
        self.special_tokens = special_tokens
        
        # Initialize with special tokens
        self.token_to_id: Dict[str, int] = dict(self.special_tokens.token_to_id)
        self.id_to_token: Dict[int, str] = dict(self.special_tokens.id_to_token)
        
        self._next_id = len(self.token_to_id)
        
    def add_token(self, token: str) -> int:
        """Adds a token to the vocabulary if it doesn't exist, returns its ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
            
        token_id = self._next_id
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self._next_id += 1
        return token_id
        
    def get_id(self, token: str) -> int:
        """Returns the ID for a token, falling back to UNK if not found."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        return self.special_tokens.get_id(self.special_tokens.config.unk_token)
        
    def get_token(self, token_id: int) -> str:
        """Returns the token string for a given ID."""
        return self.id_to_token.get(token_id, self.special_tokens.config.unk_token)
        
    def __len__(self) -> int:
        return len(self.token_to_id)
        
    def get_mapping(self) -> Dict[str, int]:
        return self.token_to_id
        
    def load_mapping(self, mapping: Dict[str, int]) -> None:
        """Loads a pre-existing mapping, replacing the current one."""
        self.token_to_id = dict(mapping)
        self.id_to_token = {v: k for k, v in mapping.items()}
        self._next_id = max(self.id_to_token.keys()) + 1 if self.id_to_token else 0
