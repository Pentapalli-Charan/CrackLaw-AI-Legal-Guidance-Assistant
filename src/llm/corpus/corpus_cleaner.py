import re
from typing import List
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.metadata import CorpusDocument

class CorpusCleaner:
    """Cleans legal text of OCR artifacts, duplicate spaces, and typical pagination."""
    
    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()
        
        # Regex Patterns
        self.duplicate_space_pattern = re.compile(r'[ \t]+')
        self.duplicate_newline_pattern = re.compile(r'\n{3,}')
        self.page_number_pattern = re.compile(r'(?i)^\s*page\s+\d+\s*of\s+\d+\s*$', re.MULTILINE)
        self.header_footer_pattern = re.compile(r'(?i)^\s*(copyright|confidential|all rights reserved).*$', re.MULTILINE)
        self.ocr_artifact_pattern = re.compile(r'[^a-zA-Z0-9\s\.,;:\'"\(\)\[\]\{\}\-—–_+=/\\|?!@#$%^&*`~<>]')

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        cleaned = text
        
        # Remove broken unicode / OCR artifacts (optional, keep common punctuation)
        # cleaned = self.ocr_artifact_pattern.sub(' ', cleaned)
        
        if self.config.remove_page_numbers:
            cleaned = self.page_number_pattern.sub('', cleaned)
            cleaned = self.header_footer_pattern.sub('', cleaned)
            
        if self.config.normalize_whitespace:
            # Replace multiple spaces with single space
            cleaned = self.duplicate_space_pattern.sub(' ', cleaned)
            # Replace 3+ newlines with 2 newlines (preserve paragraph separation)
            cleaned = self.duplicate_newline_pattern.sub('\n\n', cleaned)
            
        return cleaned.strip()

    def process(self, documents: List[CorpusDocument]) -> List[CorpusDocument]:
        """Runs the cleaner over all documents in the corpus."""
        for doc in documents:
            if doc.is_valid:
                doc.text = self.clean_text(doc.text)
                # Word count calculation after cleaning
                doc.word_count = len(doc.text.split())
        return documents
