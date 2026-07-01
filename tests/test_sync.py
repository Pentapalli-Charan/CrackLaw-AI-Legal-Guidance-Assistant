import unittest
import os
import json
import shutil
import tempfile
from src.config import Config
from src.metadata import MetadataManager
from src.registry import KnowledgeRegistry
from src.verifier import KnowledgeVerifier
from src.classification import RuleBasedClassifier
from src.sync_service import SyncService

class TestSyncService(unittest.TestCase):

    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure sandboxed directories
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.downloads_dir = os.path.join(self.temp_dir, "downloads")
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        
        self.config_data = {
            "paths": {
                "datasets_dir": self.temp_dir,
                "raw_dir": self.raw_dir,
                "processed_dir": os.path.join(self.temp_dir, "processed"),
                "cleaned_dir": os.path.join(self.temp_dir, "cleaned"),
                "chunks_dir": os.path.join(self.temp_dir, "chunks"),
                "embeddings_dir": os.path.join(self.temp_dir, "embeddings"),
                "metadata_dir": self.metadata_dir,
                "cache_dir": os.path.join(self.temp_dir, "cache"),
                "downloads_dir": self.downloads_dir,
                "logs_dir": self.logs_dir
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(self.logs_dir, "test.log")
            },
            "knowledge_acquisition": {
                "enabled_adapters": ["url", "local_folder", "indiacode", "supreme_court"],
                "retry_count": 1,
                "timeout": 5,
                "parallel_downloads": False,
                "verification": {
                    "verify_checksum": True,
                    "detect_corruption": True,
                    "skip_duplicates": True
                },
                "classification": {
                    "laws": ["act"],
                    "judgments": ["versus"],
                    "contracts": ["agreement", "contract"],
                    "legal_qa": ["question:"],
                    "legal_nlp": ["ner"],
                    "notifications": ["notification"],
                    "regulations": ["regulation"]
                }
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f)
            
        self.config = Config(self.config_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        Config._instance = None  # Reset singleton

    def test_sync_dataset_local(self):
        # Setup mock local folder to sync from
        src_folder = os.path.join(self.temp_dir, "my_src_folder")
        os.makedirs(src_folder, exist_ok=True)
        
        with open(os.path.join(src_folder, "contract_nda.txt"), "w") as f:
            f.write("MUTUAL NDA AGREEMENT\nThis agreement is entered into by the parties here...")
            
        metadata_mgr = MetadataManager(self.config)
        knowledge_reg = KnowledgeRegistry(self.config)
        verifier = KnowledgeVerifier(self.config, metadata_mgr)
        classifier = RuleBasedClassifier(self.config)
        
        sync_service = SyncService(self.config, metadata_mgr, knowledge_reg, verifier, classifier)
        
        dataset_cfg = {
            "name": "test_local_sync",
            "source_type": "local_folder",
            "source_path": src_folder,
            "version": "1.0.0",
            "license": "Custom",
            "document_types": ["contracts"],
            "supported_languages": ["en"]
        }
        
        # Run synchronization
        success = sync_service.sync_dataset(dataset_cfg)
        self.assertTrue(success)
        
        # Verify it was classified as contracts and moved to raw/contracts/
        expected_path = os.path.normpath(os.path.join(self.raw_dir, "contracts", "contract_nda.txt"))
        self.assertTrue(os.path.exists(expected_path))
        
        # Verify it was registered in metadata manager registry
        self.assertEqual(len(metadata_mgr.registry), 1)
        
        # Verify it was updated in knowledge registry
        registry_record = knowledge_reg.get_dataset("test_local_sync")
        self.assertEqual(registry_record["download_status"], "completed")
        self.assertEqual(registry_record["processing_status"], "raw")

if __name__ == "__main__":
    unittest.main()
