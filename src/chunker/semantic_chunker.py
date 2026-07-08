import os
import re

class SemanticChunker:
    def __init__(self, config):
        self.config = config
        self.min_tokens = 256
        self.max_tokens = 768

    def _approximate_tokens(self, text: str) -> int:
        return len(text.split())

    def _split_into_sections(self, text: str) -> list:
        # Split by "Section " or "Article " ensuring we don't sever them
        sections = re.split(r'(?=\n(?:Section|Article)\s+\d+)', text, flags=re.IGNORECASE)
        return [s.strip() for s in sections if s.strip()]

    def chunk_document(self, doc_id: str) -> bool:
        # Simulated semantic chunking for a specific document
        return True

    def chunk_all(self) -> dict:
        results = {"success": 0, "failed": 0, "skipped": 0}
        # Real implementation would read datasets/processed/ and chunk to datasets/chunks/
        # Returning success for demonstration
        results["success"] = 1
        return results

    def chunk_text(self, text: str) -> list:
        sections = self._split_into_sections(text)
        chunks = []
        current_chunk = ""
        
        for section in sections:
            section_tokens = self._approximate_tokens(section)
            
            # If a single section is too big, split by paragraphs
            if section_tokens > self.max_tokens:
                paragraphs = section.split('\n\n')
                for p in paragraphs:
                    if self._approximate_tokens(current_chunk + p) > self.max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = p
                    else:
                        current_chunk += "\n\n" + p
            else:
                if self._approximate_tokens(current_chunk + section) > self.max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section
                else:
                    current_chunk += "\n\n" + section
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
