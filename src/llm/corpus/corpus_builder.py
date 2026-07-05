import os
import json
import logging
from typing import List, Dict, Any, Generator
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.metadata import CorpusDocument

logger = logging.getLogger("CrackLaw.LLM.CorpusBuilder")

class CorpusBuilder:
    """Reads processed legal documents and merges them into a unified CorpusDocument list."""
    
    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()
        
    def build_corpus(self) -> List[CorpusDocument]:
        """Orchestrates reading all documents and returning a unified list."""
        corpus = []
        for doc in self._iter_processed_documents():
            corpus.append(doc)
        logger.info(f"Built corpus with {len(corpus)} documents.")
        return corpus
        
    def _iter_processed_documents(self) -> Generator[CorpusDocument, None, None]:
        """Iterates through metadata files, loads the processed text, and yields CorpusDocument objects."""
        if not os.path.exists(self.config.metadata_dir):
            logger.warning(f"Metadata directory not found: {self.config.metadata_dir}")
            return
            
        for filename in os.listdir(self.config.metadata_dir):
            if not filename.endswith(".json") or not filename.startswith("doc_"):
                continue
                
            metadata_path = os.path.join(self.config.metadata_dir, filename)
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    
                doc_id = meta.get("id")
                if not doc_id:
                    continue
                    
                doc_type = meta.get("document_type", "miscellaneous")
                processed_txt_path = meta.get("processed_file_path")
                
                # Fallback if processed_file_path is not valid or uses different os separator
                if not processed_txt_path or not os.path.exists(processed_txt_path):
                    processed_txt_path = os.path.join(self.config.processed_dir, doc_type, f"{doc_id}.txt")
                    
                if not os.path.exists(processed_txt_path):
                    logger.warning(f"Text file missing for document {doc_id} at {processed_txt_path}")
                    continue
                    
                with open(processed_txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
                # Extract hierarchical info from metadata
                doc_title = meta.get("document_title", "")
                
                act = doc_title if doc_type == "laws" else None
                judgment_title = doc_title if doc_type == "judgments" else None
                source = meta.get("source")
                language = meta.get("language", "en")
                
                # Create CorpusDocument.
                # Note: Exact sections/chapters are inherently preserved within the raw text formatting
                # of the processed document.
                corpus_doc = CorpusDocument(
                    document_id=doc_id,
                    text=text,
                    doc_type=doc_type,
                    act=act,
                    judgment_title=judgment_title,
                    source=source,
                    language=language,
                    word_count=len(text.split())
                )
                
                yield corpus_doc
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
