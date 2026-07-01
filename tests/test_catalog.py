import unittest
import os
import json
import shutil
import tempfile
from src.config import Config
from src.metadata import MetadataManager
from src.registry import KnowledgeRegistry
from src.catalog import KnowledgeCatalog

class TestCatalog(unittest.TestCase):

    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure sandboxed directories
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.cleaned_dir = os.path.join(self.temp_dir, "cleaned")
        self.chunks_dir = os.path.join(self.temp_dir, "chunks")
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        
        self.config_data = {
            "paths": {
                "datasets_dir": self.temp_dir,
                "raw_dir": self.raw_dir,
                "processed_dir": self.processed_dir,
                "cleaned_dir": self.cleaned_dir,
                "chunks_dir": self.chunks_dir,
                "embeddings_dir": os.path.join(self.temp_dir, "embeddings"),
                "metadata_dir": self.metadata_dir,
                "cache_dir": os.path.join(self.temp_dir, "cache"),
                "downloads_dir": os.path.join(self.temp_dir, "downloads"),
                "logs_dir": self.logs_dir
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(self.logs_dir, "test.log")
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f)
            
        self.config = Config(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        Config._instance = None  # Reset singleton

    def test_generate_catalog(self):
        metadata_mgr = MetadataManager(self.config)
        knowledge_reg = KnowledgeRegistry(self.config)
        
        # Create mock file in raw/laws
        laws_dir = os.path.join(self.raw_dir, "laws")
        os.makedirs(laws_dir, exist_ok=True)
        mock_file = os.path.join(laws_dir, "test_doc.txt")
        with open(mock_file, "w") as f:
            f.write("A mock legal document containing section 1 details.")
            
        # Register in main registry
        metadata_mgr.register_document(mock_file, source="test_source", doc_type="laws")
        
        # Register in knowledge registry
        dataset_cfg = {
            "name": "test_source",
            "description": "Mock source dataset",
            "source_type": "local_folder",
            "version": "1.0.0",
            "license": "Apache",
            "document_types": ["laws"],
            "supported_languages": ["en"]
        }
        knowledge_reg.register_dataset(dataset_cfg)
        knowledge_reg.update_status("test_source", download_status="completed", processing_status="raw")

        catalog_mgr = KnowledgeCatalog(self.config, metadata_mgr, knowledge_reg)
        report = catalog_mgr.generate_catalog()
        
        self.assertEqual(report["knowledge_registry_datasets"], 1)
        self.assertEqual(report["summary"]["total_documents"], 1)
        self.assertEqual(report["document_types"].get("laws"), 1)
        self.assertTrue(os.path.exists(os.path.join(self.metadata_dir, "knowledge_catalog.json")))

if __name__ == "__main__":
    unittest.main()
