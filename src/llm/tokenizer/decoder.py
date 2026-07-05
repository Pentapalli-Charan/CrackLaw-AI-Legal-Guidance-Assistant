from typing import List
from src.llm.tokenizer.vocabulary import Vocabulary

class Decoder:
    """Converts sequences of Token IDs back into raw text."""
    
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        
    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into text."""
        if not token_ids:
            return ""
            
        text_parts = []
        for token_id in token_ids:
            token = self.vocab.get_token(token_id)
            
            # If the token is a special token, we just append it as is.
            # BPE tokens are strings, so we just concatenate them.
            text_parts.append(token)
            
        return "".join(text_parts)
