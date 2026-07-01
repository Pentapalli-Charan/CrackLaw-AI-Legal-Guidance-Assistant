import unittest
import os
import json
import shutil
import tempfile
from src.config import Config
from src.adapters import AdapterFactory, URLAdapter, LocalFolderAdapter, IndiaCodeAdapter, SupremeCourtAdapter

class TestAdapters(unittest.TestCase):
    
    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure sandboxed directories
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.downloads_dir = os.path.join(self.temp_dir, "downloads")
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
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
                "cache_dir": self.cache_dir,
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

    def test_adapter_factory(self):
        self.assertIsInstance(AdapterFactory.get_adapter("url"), URLAdapter)
        self.assertIsInstance(AdapterFactory.get_adapter("local_folder"), LocalFolderAdapter)
        self.assertIsInstance(AdapterFactory.get_adapter("indiacode"), IndiaCodeAdapter)
        self.assertIsInstance(AdapterFactory.get_adapter("supreme_court"), SupremeCourtAdapter)
        
        with self.assertRaises(ValueError):
            AdapterFactory.get_adapter("invalid_adapter_name")

    def test_local_folder_adapter(self):
        src_dir = os.path.join(self.temp_dir, "local_src")
        os.makedirs(src_dir, exist_ok=True)
        
        file1 = os.path.join(src_dir, "doc1.txt")
        file2 = os.path.join(src_dir, "doc2.html")
        with open(file1, "w") as f:
            f.write("Some mock law details section 1.")
        with open(file2, "w") as f:
            f.write("<html><body>Mutual Agreement</body></html>")
            
        adapter = LocalFolderAdapter(self.config)
        dataset_cfg = {
            "name": "test_local",
            "source_path": src_dir
        }
        
        # Test synchronization to specific target downloads staging dir
        synced_files = adapter.sync(dataset_cfg, self.downloads_dir)
        self.assertEqual(len(synced_files), 2)
        
        copied_file1 = os.path.join(self.downloads_dir, "doc1.txt")
        copied_file2 = os.path.join(self.downloads_dir, "doc2.html")
        self.assertTrue(os.path.exists(copied_file1))
        self.assertTrue(os.path.exists(copied_file2))
        
        # Test check updates
        # Ensure test_local folder under self.config.downloads_dir has files matching source size
        local_target_dir = os.path.join(self.config.downloads_dir, "test_local")
        os.makedirs(local_target_dir, exist_ok=True)
        shutil.copy2(file1, os.path.join(local_target_dir, "doc1.txt"))
        shutil.copy2(file2, os.path.join(local_target_dir, "doc2.html"))
        
        self.assertFalse(adapter.check_updates(dataset_cfg))
        
        # Add a new file to source
        with open(os.path.join(src_dir, "doc3.txt"), "w") as f:
            f.write("New update.")
        self.assertTrue(adapter.check_updates(dataset_cfg))

    def test_indiacode_adapter_fallback(self):
        adapter = IndiaCodeAdapter(self.config)
        dataset_cfg = {
            "name": "test_indiacode",
            "query": "Taxation",
            "max_docs": 1
        }
        
        synced = adapter.sync(dataset_cfg, self.downloads_dir)
        self.assertTrue(len(synced) >= 1)
        self.assertTrue(os.path.exists(synced[0]))

    def test_supreme_court_adapter_fallback(self):
        adapter = SupremeCourtAdapter(self.config)
        dataset_cfg = {
            "name": "test_sc",
            "year": 2026
        }
        
        synced = adapter.sync(dataset_cfg, self.downloads_dir)
        self.assertEqual(len(synced), 1)
        self.assertTrue(os.path.exists(synced[0]))

if __name__ == "__main__":
    unittest.main()
