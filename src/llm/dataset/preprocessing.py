from typing import List, Optional
import logging

logger = logging.getLogger("CrackLaw.LLM.DatasetPreprocessor")

class DataPreprocessor:
    """Preprocesses text before it enters the Dataset/Tokenizer layer."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes whitespace and standardizes basic text formats."""
        if not text:
            return ""
        # Collapse multiple spaces
        text = " ".join(text.split())
        return text.strip()
        
    @staticmethod
    def is_valid_sample(text: str, min_chars: int = 10) -> bool:
        """Determines if a sample is valid to be included in the dataset."""
        if not text:
            return False
        if len(text.strip()) < min_chars:
            return False
        return True
