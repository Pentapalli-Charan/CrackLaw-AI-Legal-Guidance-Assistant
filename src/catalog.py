import os
import json
import time
from typing import Optional, Dict, Any
from src.config import Config
from src.metadata import MetadataManager
from src.registry import KnowledgeRegistry

class KnowledgeCatalog:
    """Compiles metrics across both source registries and file nodes to output a dataset catalog."""

    def __init__(
        self,
        config: Optional[Config] = None,
        metadata_manager: Optional[MetadataManager] = None,
        knowledge_registry: Optional[KnowledgeRegistry] = None
    ):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.knowledge_reg = knowledge_registry or KnowledgeRegistry(self.config)
        self.catalog_path = os.path.normpath(
            os.path.join(self.config.metadata_dir, "knowledge_catalog.json")
        )

    def _get_dir_size_mb(self, directory: str) -> float:
        """Helper to calculate total size of a directory recursively in MB."""
        total_size = 0
        if not os.path.exists(directory):
            return 0.0
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return round(total_size / (1024 * 1024), 2)

    def generate_catalog(self) -> Dict[str, Any]:
        """Calculates and writes summary statistics for all datasets."""
        # 1. Fetch document and registry counts
        stats = self.metadata_mgr.generate_statistics()
        datasets = self.knowledge_reg.list_datasets()
        
        # 2. Count failed downloads and check last sync
        failed_downloads = 0
        last_sync = None
        
        for d in datasets:
            if d.get("download_status") == "failed":
                failed_downloads += 1
            l_updated = d.get("last_updated")
            if l_updated:
                if not last_sync or l_updated > last_sync:
                    last_sync = l_updated

        # 3. Determine duplicate count from individual metadata files
        duplicate_count = 0
        for doc_id, meta in list(self.metadata_mgr.registry.items()):
            indiv = self.metadata_mgr.get_document_metadata(doc_id)
            if indiv and "duplicate" in str(indiv.get("error_log", "")).lower():
                duplicate_count += 1

        # 4. Check folder storage sizes
        raw_size = self._get_dir_size_mb(self.config.raw_dir)
        processed_size = self._get_dir_size_mb(self.config.processed_dir)
        cleaned_size = self._get_dir_size_mb(self.config.cleaned_dir)
        chunks_size = self._get_dir_size_mb(self.config.chunks_dir)
        total_size = round(raw_size + processed_size + cleaned_size + chunks_size, 2)

        catalog_data = {
            "last_synchronization": last_sync,
            "summary": {
                "total_documents": stats["total_documents"],
                "total_chunks": stats["total_chunks"],
                "total_storage_size_mb": total_size,
                "raw_storage_mb": raw_size,
                "processed_storage_mb": processed_size,
                "cleaned_storage_mb": cleaned_size,
                "chunks_storage_mb": chunks_size,
                "duplicate_count": duplicate_count,
                "failed_downloads": failed_downloads
            },
            "document_types": stats["document_types"],
            "processing_progress": stats["status_distribution"],
            "sources": stats["sources"],
            "knowledge_registry_datasets": len(datasets)
        }

        # Save to JSON
        os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog_data, f, indent=2)

        return catalog_data
