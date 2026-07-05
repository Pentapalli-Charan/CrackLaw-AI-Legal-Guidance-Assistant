import hashlib
from typing import List
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.metadata import CorpusDocument

class CorpusValidator:
    """Validates documents for emptiness, duplicates, and encoding issues."""
    
    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()
        self.seen_hashes = set()
        
    def _compute_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()

    def validate_document(self, doc: CorpusDocument) -> None:
        if not doc.text or not doc.text.strip():
            doc.is_valid = False
            doc.validation_errors.append("Empty document")
            return
            
        if doc.word_count < self.config.min_word_count:
            doc.is_valid = False
            doc.validation_errors.append(f"Document too short ({doc.word_count} words)")
            return
            
        doc_hash = self._compute_hash(doc.text)
        if doc_hash in self.seen_hashes:
            doc.is_valid = False
            doc.validation_errors.append("Duplicate document")
            return
            
        # Encoding issues check
        if '\ufffd' in doc.text:  # Replacement character
            doc.validation_errors.append("Encoding issues detected")
            # We don't mark invalid, but we flag it.
            
        self.seen_hashes.add(doc_hash)

    def process(self, documents: List[CorpusDocument]) -> List[CorpusDocument]:
        """Validates a list of documents and optionally filters them."""
        self.seen_hashes.clear()
        
        valid_docs = []
        for doc in documents:
            self.validate_document(doc)
            if doc.is_valid or not self.config.drop_empty:
                valid_docs.append(doc)
                
        return valid_docs
