import os
import json
import csv
import logging
from typing import List
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.metadata import CorpusDocument

logger = logging.getLogger("CrackLaw.LLM.CorpusExporter")

class CorpusExporter:
    """Exports the cleaned corpus to standard LLM training formats (JSONL, TXT, CSV)."""
    
    def __init__(self, config: CorpusConfig = None):
        self.config = config or CorpusConfig()

    def export_all(self, documents: List[CorpusDocument], base_filename: str = "cracklaw_corpus") -> None:
        """Exports corpus to all supported formats."""
        self.export_jsonl(documents, f"{base_filename}.jsonl")
        self.export_txt(documents, f"{base_filename}.txt")
        self.export_csv(documents, f"{base_filename}.csv")
        logger.info(f"Exported corpus to {self.config.corpus_out_dir} in JSONL, TXT, and CSV formats.")

    def export_jsonl(self, documents: List[CorpusDocument], filename: str) -> str:
        """Exports to JSONL (one JSON object per line)."""
        filepath = os.path.join(self.config.corpus_out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in documents:
                # We only need to export the relevant fields for training
                export_dict = {
                    "text": doc.text,
                    "metadata": {
                        "document_id": doc.document_id,
                        "doc_type": doc.doc_type,
                        "act": doc.act,
                        "judgment_title": doc.judgment_title,
                        "source": doc.source,
                        "language": doc.language
                    }
                }
                f.write(json.dumps(export_dict, ensure_ascii=False) + "\n")
        return filepath

    def export_txt(self, documents: List[CorpusDocument], filename: str) -> str:
        """Exports to a single continuous TXT file (useful for simple language modeling)."""
        filepath = os.path.join(self.config.corpus_out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in documents:
                # Separate documents with a clear delimiter
                f.write(f"--- BEGIN DOCUMENT: {doc.document_id} ---\n")
                f.write(doc.text)
                f.write(f"\n--- END DOCUMENT ---\n\n")
        return filepath

    def export_csv(self, documents: List[CorpusDocument], filename: str) -> str:
        """Exports to CSV for easy inspection in pandas or spreadsheet tools."""
        filepath = os.path.join(self.config.corpus_out_dir, filename)
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["document_id", "doc_type", "act", "judgment_title", "source", "text"])
            
            for doc in documents:
                writer.writerow([
                    doc.document_id,
                    doc.doc_type,
                    doc.act or "",
                    doc.judgment_title or "",
                    doc.source or "",
                    doc.text
                ])
        return filepath
