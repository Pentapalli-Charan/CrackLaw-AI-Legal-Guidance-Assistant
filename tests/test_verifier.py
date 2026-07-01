import unittest
import os
import shutil
import tempfile
from src.config import Config
from src.metadata import MetadataManager
from src.verifier import KnowledgeVerifier

class TestVerifier(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        self.config._data["paths"]["metadata_dir"] = self.metadata_dir
        self.config._data["paths"]["raw_dir"] = self.raw_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_verify_corrupt_files(self):
        metadata_mgr = MetadataManager(self.config)
        verifier = KnowledgeVerifier(self.config, metadata_mgr)
        
        # 1. Test empty file corruption
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, "w") as f:
            pass # write nothing
            
        report = verifier.verify_file(empty_file)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("empty" in e.lower() for e in report["errors"]))
        
        # 2. Test invalid JSON corruption
        bad_json = os.path.join(self.temp_dir, "bad.json")
        with open(bad_json, "w") as f:
            f.write("{invalid json structure")
            
        report = verifier.verify_file(bad_json)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("parsing error" in e.lower() for e in report["errors"]))

    def test_verify_duplicates(self):
        metadata_mgr = MetadataManager(self.config)
        verifier = KnowledgeVerifier(self.config, metadata_mgr)
        
        # Register a file in metadata manager
        orig_file = os.path.join(self.raw_dir, "original.txt")
        with open(orig_file, "w") as f:
            f.write("Highly specific unique document content text.")
        metadata_mgr.register_document(orig_file, source="test", doc_type="laws")
        
        # Attempt to verify a duplicate file (same content)
        dup_file = os.path.join(self.temp_dir, "duplicate.txt")
        with open(dup_file, "w") as f:
            f.write("Highly specific unique document content text.")
            
        report = verifier.verify_file(dup_file)
        self.assertFalse(report["is_valid"])
        self.assertIsNotNone(report["duplicate_doc_id"])
        self.assertTrue(any("duplicate" in e.lower() for e in report["errors"]))

    def test_verify_incomplete_downloads(self):
        metadata_mgr = MetadataManager(self.config)
        verifier = KnowledgeVerifier(self.config, metadata_mgr)
        
        target_file = os.path.join(self.temp_dir, "file.txt")
        part_file = target_file + ".part"
        
        with open(target_file, "w") as f:
            f.write("Some half finished download content.")
        with open(part_file, "w") as f:
            f.write("Part download data.")
            
        report = verifier.verify_file(target_file)
        self.assertFalse(report["is_valid"])
        self.assertTrue(any("incomplete" in e.lower() for e in report["errors"]))

if __name__ == "__main__":
    unittest.main()
