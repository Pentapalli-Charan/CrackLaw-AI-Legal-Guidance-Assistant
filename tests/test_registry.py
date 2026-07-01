import unittest
import os
import json
import shutil
import tempfile
from src.config import Config
from src.registry import KnowledgeRegistry

class TestRegistry(unittest.TestCase):

    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure sandboxed directories
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        
        self.config_data = {
            "paths": {
                "datasets_dir": self.temp_dir,
                "raw_dir": os.path.join(self.temp_dir, "raw"),
                "processed_dir": os.path.join(self.temp_dir, "processed"),
                "cleaned_dir": os.path.join(self.temp_dir, "cleaned"),
                "chunks_dir": os.path.join(self.temp_dir, "chunks"),
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

    def test_registry_registration_and_status(self):
        registry_mgr = KnowledgeRegistry(self.config)
        
        dataset_cfg = {
            "name": "test_dataset_registry",
            "description": "Mock description",
            "source_type": "url",
            "version": "2.1.0",
            "license": "MIT",
            "document_types": ["contracts"],
            "supported_languages": ["en"]
        }
        
        # 1. Register
        dataset = registry_mgr.register_dataset(dataset_cfg)
        self.assertEqual(dataset["name"], "test_dataset_registry")
        self.assertEqual(dataset["version"], "2.1.0")
        self.assertEqual(dataset["download_status"], "pending")
        
        # 2. Check individual registry retrieval
        retrieved = registry_mgr.get_dataset("test_dataset_registry")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["license"], "MIT")
        
        # 3. Update Status
        registry_mgr.update_status("test_dataset_registry", download_status="completed", processing_status="processed")
        updated = registry_mgr.get_dataset("test_dataset_registry")
        self.assertEqual(updated["download_status"], "completed")
        self.assertEqual(updated["processing_status"], "processed")
        self.assertIsNotNone(updated["last_updated"])
        
        # 4. Check list
        datasets_list = registry_mgr.list_datasets()
        self.assertEqual(len(datasets_list), 1)
        self.assertEqual(datasets_list[0]["name"], "test_dataset_registry")
        
        # 5. Check persistence (create new manager instance pointing to same file)
        new_registry_mgr = KnowledgeRegistry(self.config)
        self.assertEqual(len(new_registry_mgr.list_datasets()), 1)
        self.assertEqual(new_registry_mgr.get_dataset("test_dataset_registry")["download_status"], "completed")

if __name__ == "__main__":
    unittest.main()
