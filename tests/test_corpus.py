import unittest
import os
import tempfile
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.metadata import CorpusDocument
from src.llm.corpus.corpus_cleaner import CorpusCleaner
from src.llm.corpus.corpus_validator import CorpusValidator
from src.llm.corpus.corpus_statistics import CorpusStatistics
from src.llm.corpus.corpus_exporter import CorpusExporter

class TestCorpusPipeline(unittest.TestCase):
    
    def setUp(self):
        self.config = CorpusConfig()
        
    def test_cleaner_removes_artifacts(self):
        cleaner = CorpusCleaner(self.config)
        
        raw_text = "CHAPTER I\n\n\n\nSection 1\n   This   has   extra spaces.  \nPage 1 of 10\n"
        doc = CorpusDocument(document_id="test1", text=raw_text, doc_type="laws")
        
        cleaned_docs = cleaner.process([doc])
        cleaned_text = cleaned_docs[0].text
        
        self.assertNotIn("Page 1 of 10", cleaned_text, "Pagination not removed")
        self.assertNotIn("   ", cleaned_text, "Duplicate spaces not removed")
        self.assertNotIn("\n\n\n", cleaned_text, "Duplicate newlines not normalized")
        self.assertIn("CHAPTER I", cleaned_text, "Valid content was removed")
        
    def test_validator_detects_empty_and_duplicates(self):
        validator = CorpusValidator(self.config)
        
        doc1 = CorpusDocument(document_id="d1", text="This is a valid legal document with enough words to pass validation.", doc_type="laws", word_count=12)
        doc2 = CorpusDocument(document_id="d2", text="Too short", doc_type="laws", word_count=2)
        doc3 = CorpusDocument(document_id="d3", text="This is a valid legal document with enough words to pass validation.", doc_type="laws", word_count=12)  # Duplicate
        
        valid_docs = validator.process([doc1, doc2, doc3])
        
        self.assertEqual(len(valid_docs), 1, "Only one document should pass validation")
        self.assertEqual(valid_docs[0].document_id, "d1")
        self.assertFalse(doc2.is_valid, "Short document should be invalid")
        self.assertFalse(doc3.is_valid, "Duplicate document should be invalid")
        
    def test_statistics_generation(self):
        stats_engine = CorpusStatistics()
        doc1 = CorpusDocument(document_id="d1", text="Hello world", doc_type="laws", word_count=2)
        doc2 = CorpusDocument(document_id="d2", text="Judgment text here", doc_type="judgments", word_count=3)
        
        report = stats_engine.generate_report([doc1, doc2])
        self.assertEqual(report["total_documents"], 2)
        self.assertEqual(report["number_of_acts"], 1)
        self.assertEqual(report["number_of_judgments"], 1)
        self.assertEqual(report["total_words"], 5)
        self.assertIn("largest_document", report)
        
    def test_exporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CorpusConfig(corpus_out_dir=tmpdir)
            exporter = CorpusExporter(config)
            
            doc1 = CorpusDocument(document_id="d1", text="Hello world", doc_type="laws", word_count=2)
            exporter.export_all([doc1], "test_corpus")
            
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_corpus.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_corpus.txt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_corpus.csv")))

if __name__ == '__main__':
    unittest.main()
