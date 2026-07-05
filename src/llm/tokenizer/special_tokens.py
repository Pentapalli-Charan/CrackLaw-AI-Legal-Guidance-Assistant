from typing import List, Dict
from src.llm.tokenizer.config import TokenizerConfig

class SpecialTokens:
    """Manages special tokens and their reserved IDs in the vocabulary."""
    
    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        
        self.special_tokens_list = [
            self.config.pad_token,
            self.config.unk_token,
            self.config.bos_token,
            self.config.eos_token,
            self.config.mask_token,
            self.config.cls_token,
            self.config.sep_token
        ]
        
        # Mapping token -> reserved ID
        self.token_to_id: Dict[str, int] = {
            tok: i for i, tok in enumerate(self.special_tokens_list)
        }
        self.id_to_token: Dict[int, str] = {
            i: tok for tok, i in self.token_to_id.items()
        }
        
    def get_num_special_tokens(self) -> int:
        return len(self.special_tokens_list)
        
    def is_special(self, token: str) -> bool:
        return token in self.token_to_id
        
    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[self.config.unk_token])
