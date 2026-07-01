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

class TestImports(unittest.TestCase):

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

    def test_import_local_folder_functionality(self):
        # 1. Setup folders and managers
        src_folder = os.path.join(self.temp_dir, "import_source")
        os.makedirs(src_folder, exist_ok=True)
        
        metadata_mgr = MetadataManager(self.config)
        knowledge_reg = KnowledgeRegistry(self.config)
        verifier = KnowledgeVerifier(self.config, metadata_mgr)
        classifier = RuleBasedClassifier(self.config)
        
        sync_service = SyncService(self.config, metadata_mgr, knowledge_reg, verifier, classifier)
        
        # 2. Write mock files to source directory
        # Valid contract file
        with open(os.path.join(src_folder, "lease_agreement.txt"), "w") as f:
            f.write("LEASE AGREEMENT\nThis lease contract entered into by the parties landlord and tenant...")
            
        # Empty file (should fail verification)
        with open(os.path.join(src_folder, "empty.txt"), "w") as f:
            pass
            
        # 3. Perform manual import
        results = sync_service.import_local_folder(src_folder, default_category="miscellaneous")
        
        self.assertEqual(results["total_files"], 2)
        self.assertEqual(results["imported"], 1)
        self.assertEqual(results["failed_verification"], 1)
        self.assertEqual(results["skipped_duplicates"], 0)
        
        # Verify the contract was copied into raw/contracts/
        expected_contract_path = os.path.normpath(os.path.join(self.raw_dir, "contracts", "lease_agreement.txt"))
        self.assertTrue(os.path.exists(expected_contract_path))
        
        # 4. Try importing same folder again (should skip duplicate as duplicate_count increases)
        results_again = sync_service.import_local_folder(src_folder, default_category="miscellaneous")
        self.assertEqual(results_again["imported"], 0)
        # The contract is skipped as duplicate, empty fails verification
        self.assertEqual(results_again["skipped_duplicates"], 1)
        self.assertEqual(results_again["failed_verification"], 1)

if __name__ == "__main__":
    unittest.main()
