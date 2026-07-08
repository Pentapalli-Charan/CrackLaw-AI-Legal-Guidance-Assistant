import os
import hashlib
import json
from src.ingestion.parsers import DocumentParser
from src.ingestion.cleaner import DocumentCleaner, MetadataExtractor

class IngestionPipeline:
    def __init__(self, config):
        self.config = config
        self.processed_hashes = set()

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def ingest_file(self, file_path: str, output_dir: str) -> bool:
        try:
            raw_text = DocumentParser.parse(file_path)
            if not raw_text.strip():
                return False
                
            # Deduplication
            content_hash = self._get_hash(raw_text)
            if content_hash in self.processed_hashes:
                return False
            self.processed_hashes.add(content_hash)
            
            cleaned_text = DocumentCleaner.clean_text(raw_text)
            metadata = MetadataExtractor.extract(cleaned_text, os.path.basename(file_path))
            
            # Save into category folder
            category_dir = os.path.join(output_dir, metadata["category"].replace(" ", "_"))
            os.makedirs(category_dir, exist_ok=True)
            
            output_file = os.path.join(category_dir, f"{content_hash}.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                record = {
                    "text": cleaned_text,
                    "metadata": metadata
                }
                f.write(json.dumps(record) + "\n")
            return True
        except Exception:
            return False

    def ingest_all(self) -> dict:
        results = {"success": 0, "failed": 0, "skipped": 0}
        raw_dir = self.config.get("paths", {}).get("raw_dir", "datasets/raw")
        processed_dir = self.config.get("paths", {}).get("processed_dir", "datasets/processed")
        
        if not os.path.exists(raw_dir):
            return results
            
        for root, _, files in os.walk(raw_dir):
            for file in files:
                path = os.path.join(root, file)
                if self.ingest_file(path, processed_dir):
                    results["success"] += 1
                else:
                    results["skipped"] += 1
        return results
