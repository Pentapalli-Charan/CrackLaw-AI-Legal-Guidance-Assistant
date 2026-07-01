import os
import json
import time
from typing import Any, Dict, Optional
from src.config import Config
from src.utils import calculate_checksum

class MetadataManager:
    """Manages central document registry and individual document metadata records."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.registry_path = os.path.join(self.config.metadata_dir, "dataset_registry.json")
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Loads the central dataset registry JSON file."""
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            # If corrupt, backup and start fresh, or raise
            backup_path = f"{self.registry_path}.bak.{int(time.time())}"
            if os.path.exists(self.registry_path):
                os.rename(self.registry_path, backup_path)
            # Initialize empty
            return {}

    def save_registry(self) -> None:
        """Saves the central registry JSON file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)

    def generate_doc_id(self, checksum: str) -> str:
        """Generates a unique document ID based on its checksum."""
        # Use first 16 characters of the checksum as a clean ID
        return f"doc_{checksum[:16]}"

    def check_duplicate(self, file_path: str) -> Optional[str]:
        """Checks if a file already exists in the registry based on checksum."""
        try:
            checksum = calculate_checksum(file_path)
            for doc_id, meta in self.registry.items():
                if meta.get("checksum") == checksum:
                    return doc_id
        except Exception:
            pass
        return None

    def register_document(
        self,
        file_path: str,
        source: str,
        doc_type: str,
        language: str = "en",
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Registers a document, creating individual metadata and updating registry."""
        checksum = calculate_checksum(file_path)
        doc_id = self.generate_doc_id(checksum)
        
        # Check if already registered
        if doc_id in self.registry:
            return self.registry[doc_id]

        filename = os.path.basename(file_path)
        size_bytes = os.path.getsize(file_path)
        
        metadata = {
            "id": doc_id,
            "original_filename": filename,
            "file_path": file_path,
            "document_title": title or os.path.splitext(filename)[0],
            "source": source,
            "date_added": time.strftime("%Y-%m-%d %H:%M:%S"),
            "document_type": doc_type,
            "language": language,
            "checksum": checksum,
            "size_bytes": size_bytes,
            "processing_status": "raw",  # raw, processed, cleaned, chunked, failed
            "chunk_count": 0,
            "embedding_status": "pending",  # pending, completed, failed
            "error_log": None
        }

        # Save individual metadata file
        meta_file_path = os.path.join(self.config.metadata_dir, f"{doc_id}.json")
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update registry
        self.registry[doc_id] = {
            "id": doc_id,
            "original_filename": filename,
            "document_title": metadata["document_title"],
            "source": source,
            "document_type": doc_type,
            "checksum": checksum,
            "processing_status": "raw",
            "chunk_count": 0,
            "embedding_status": "pending",
            "date_added": metadata["date_added"]
        }
        self.save_registry()
        
        return metadata

    def update_metadata(self, doc_id: str, updates: Dict[str, Any]) -> None:
        """Updates metadata fields for a document in both individual file and registry."""
        meta_file_path = os.path.join(self.config.metadata_dir, f"{doc_id}.json")
        
        # Load individual file
        metadata = {}
        if os.path.exists(meta_file_path):
            with open(meta_file_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
        # Merge updates
        metadata.update(updates)
        
        # Save individual file
        with open(meta_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        # Update registry (only keep subset of fields in central registry to keep it light)
        if doc_id in self.registry:
            for k in ["original_filename", "document_title", "source", "document_type", "checksum", "processing_status", "chunk_count", "embedding_status"]:
                if k in updates:
                    self.registry[doc_id][k] = updates[k]
            self.save_registry()

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Loads and returns metadata for a single document."""
        meta_file_path = os.path.join(self.config.metadata_dir, f"{doc_id}.json")
        if not os.path.exists(meta_file_path):
            return None
        with open(meta_file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_statistics(self) -> Dict[str, Any]:
        """Generates comprehensive statistics over all registered datasets."""
        total_docs = len(self.registry)
        status_counts = {}
        type_counts = {}
        source_counts = {}
        total_chunks = 0
        
        for doc_id, meta in self.registry.items():
            status = meta.get("processing_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            doc_type = meta.get("document_type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            source = meta.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            
            total_chunks += meta.get("chunk_count", 0)

        # Check actual files sizes from individual metadata
        total_bytes = 0
        for doc_id in self.registry:
            indiv = self.get_document_metadata(doc_id)
            if indiv:
                total_bytes += indiv.get("size_bytes", 0)
                
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "status_distribution": status_counts,
            "document_types": type_counts,
            "sources": source_counts
        }
