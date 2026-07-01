import unittest
import os
import json
import shutil
import tempfile
from unittest.mock import MagicMock
from src.config import Config
from src.retrieval.exceptions import RetrievalError
from src.retrieval.search_types import SearchMode
from src.retrieval.search_result import SearchResultItem
from src.retrieval.embedding_engine import EmbeddingEngine
from src.retrieval.vector_store import VectorStore
from src.retrieval.metadata_filter import MetadataFilter
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.semantic_search import SemanticSearch
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.retrieval_service import RetrievalService

class TestRetrievalEngine(unittest.TestCase):

    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure sandboxed directories
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        self.chunks_dir = os.path.join(self.temp_dir, "chunks")
        self.logs_dir = os.path.join(self.temp_dir, "logs")
        
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.config_data = {
            "paths": {
                "datasets_dir": self.temp_dir,
                "raw_dir": os.path.join(self.temp_dir, "raw"),
                "processed_dir": os.path.join(self.temp_dir, "processed"),
                "cleaned_dir": os.path.join(self.temp_dir, "cleaned"),
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
            },
            "retrieval": {
                "embedding_model": "bge",
                "top_k": 3,
                "similarity_threshold": 0.2,
                "hybrid_mode": "hybrid",
                "hybrid_weights": {
                    "semantic": 0.6,
                    "keyword": 0.4
                },
                "batch_size": 2,
                "use_gpu": False,
                "cache": {
                    "embedding_cache_size": 100,
                    "query_cache_size": 50
                },
                "reranking": {
                    "enabled": False,
                    "model_name": "mock-reranker",
                    "rerank_depth": 5
                }
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f)
            
        self.config = Config(self.config_path)

        # Mock the underlying embedding model to avoid downloading parameters
        self.mock_model = MagicMock()
        self.mock_model.dimension = 4
        self.mock_model.embed_query.return_value = [0.25, 0.25, 0.25, 0.25]
        self.mock_model.embed_documents.return_value = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0]
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        Config._instance = None  # Reset singleton

    def _create_engine_with_mocks(self):
        engine = EmbeddingEngine(self.config)
        engine.model = self.mock_model
        return engine

    def test_embedding_engine_mock(self):
        engine = self._create_engine_with_mocks()
        
        # Test Query
        q_emb = engine.embed_query("search terms")
        self.assertEqual(len(q_emb), 4)
        self.assertEqual(q_emb, [0.25, 0.25, 0.25, 0.25])
        
        # Test Cache Hit
        q_emb_cached = engine.embed_query("search terms")
        self.assertEqual(q_emb_cached, q_emb)
        # Verify mock model embed_query was only called once due to cache
        self.mock_model.embed_query.assert_called_once()
        
        # Test Batch Documents
        docs = ["Environmental laws", "Judicial orders", "NDA Contracts"]
        d_embs = engine.embed_documents(docs)
        self.assertEqual(len(d_embs), 3)
        self.assertEqual(d_embs[0], [1.0, 0.0, 0.0, 0.0])

    def test_vector_store_operations(self):
        # Force fallback local numpy database
        store = VectorStore(self.config, dimension=4)
        store.use_fallback = True
        
        # Insert mock chunks
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "doc1",
                "text": "The environmental protection guidelines section 5.",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "act": "Environmental Act",
                "section": "5",
                "language": "en"
            },
            {
                "chunk_id": "c2",
                "document_id": "doc1",
                "text": "Mutual agreement contract and NDA section 10.",
                "embedding": [0.0, 1.0, 0.0, 0.0],
                "act": "NDA Contract",
                "section": "10",
                "language": "en"
            }
        ]
        store.insert_chunks_batch(chunks)
        self.assertEqual(len(store._fallback_db), 2)
        
        # Test semantic search on vector store
        query_vec = [1.0, 0.0, 0.0, 0.0]
        results = store.search_vector(query_vec, top_k=2, min_similarity=0.0)
        self.assertEqual(len(results), 2)
        
        # Most similar should be c1 (cosine similarity of 1.0)
        self.assertEqual(results[0].chunk_id, "c1")
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertEqual(results[0].citation, "Environmental Act, Section 5")
        
        # Least similar should be c2 (orthogonal vector, cosine similarity of 0.0)
        # Should be filtered out if we set threshold above 0.0
        filtered = store.search_vector(query_vec, top_k=2, min_similarity=0.5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, "c1")

    def test_metadata_filter(self):
        raw_filters = {
            "ACT": "Taxation Act",
            "section": "12",
            "unsupported_field": "ignore_me",
            "Language": None
        }
        cleaned = MetadataFilter.clean_filters(raw_filters)
        self.assertIn("act", cleaned)
        self.assertIn("section", cleaned)
        self.assertNotIn("unsupported_field", cleaned)
        self.assertNotIn("language", cleaned)
        self.assertEqual(cleaned["act"], "Taxation Act")

    def test_keyword_search_bm25(self):
        store = VectorStore(self.config, dimension=4)
        store.use_fallback = True
        
        chunks = [
            {"chunk_id": "k1", "document_id": "d1", "text": "Environmental Protection Guidelines", "act": "Env"},
            {"chunk_id": "k2", "document_id": "d1", "text": "Mutual Agreement Lease NDA Contract", "act": "Nda"},
            {"chunk_id": "k3", "document_id": "d2", "text": "unrelated dummy text document", "act": "Other"},
            {"chunk_id": "k4", "document_id": "d2", "text": "completely different content text block", "act": "Other"}
        ]
        
        keyword_search = KeywordSearch(self.config, store)
        
        # Query matching k2
        results = keyword_search.search_local_bm25("NDA Contract", chunks, top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0].chunk_id, "k2")
        self.assertTrue(results[0].score > 0.0)

    def test_hybrid_search_fusion(self):
        sem_results = [
            SearchResultItem(chunk_id="ch1", document_id="doc1", text="text1", score=0.9, semantic_score=0.9),
            SearchResultItem(chunk_id="ch2", document_id="doc1", text="text2", score=0.8, semantic_score=0.8)
        ]
        key_results = [
            SearchResultItem(chunk_id="ch2", document_id="doc1", text="text2", score=12.0, keyword_score=12.0),
            SearchResultItem(chunk_id="ch3", document_id="doc2", text="text3", score=5.0, keyword_score=5.0)
        ]
        
        mock_semantic = MagicMock()
        mock_keyword = MagicMock()
        hybrid = HybridSearch(self.config, semantic_search=mock_semantic, keyword_search=mock_keyword)
        
        # Test RRF
        rrf_fused = hybrid._reciprocal_rank_fusion(sem_results, key_results, k=60)
        # ch2 should rank first as it appears in both lists
        self.assertEqual(rrf_fused[0].chunk_id, "ch2")
        self.assertEqual(len(rrf_fused), 3)
        
        # Test Linear Combination
        linear_fused = hybrid._linear_fusion(sem_results, key_results)
        self.assertEqual(len(linear_fused), 3)

    def test_context_builder(self):
        items = [
            SearchResultItem(chunk_id="x1", document_id="d1", text="Guideline 1", citation="Act A, Sec 1"),
            SearchResultItem(chunk_id="x2", document_id="d1", text="Guideline 2", citation="Act A, Sec 2"),
            SearchResultItem(chunk_id="x1", document_id="d1", text="Guideline 1", citation="Act A, Sec 1") # duplicate
        ]
        
        builder = ContextBuilder(self.config)
        
        # Test context formatting and deduplication
        ctx = builder.build_context(items, token_budget=1000)
        self.assertIn("Guideline 1", ctx)
        self.assertIn("Guideline 2", ctx)
        # Check deduplication: Guideline 1 should only occur once
        self.assertEqual(ctx.count("Guideline 1"), 1)
        
        # Test token budget truncation
        ctx_small = builder.build_context(items, token_budget=10) # very small budget
        self.assertEqual(len(ctx_small), 0)

    def test_unified_retrieval_service(self):
        # Write a mock chunk JSON file
        chunk_file = os.path.join(self.chunks_dir, "test_doc_chunks.json")
        mock_chunks = {
            "chunks": [
                {
                    "chunk_id": "test_chunk_1",
                    "text": "Taxation assessment guidelines.",
                    "metadata": {"act": "Tax Act", "section": "12"}
                }
            ]
        }
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(mock_chunks, f)

        # Initialize Service
        emb_engine = self._create_engine_with_mocks()
        
        # Setup vector store mock fallback
        vector_store = VectorStore(self.config, dimension=4)
        vector_store.use_fallback = True
        
        service = RetrievalService(
            config=self.config,
            embedding_engine=emb_engine,
            vector_store=vector_store
        )
        
        # Test index_document
        num_indexed = service.index_document("test_doc")
        self.assertEqual(num_indexed, 1)
        self.assertEqual(len(vector_store._fallback_db), 1)
        
        # Test retrieve
        response = service.retrieve("Taxation query", mode=SearchMode.SEMANTIC)
        self.assertEqual(response.query, "Taxation query")
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].chunk_id, "test_chunk_1")
        self.assertIn("Taxation assessment", response.context)
        self.assertTrue(response.latency_ms > 0)
        
        # Query Cache hit check
        response2 = service.retrieve("Taxation query", mode=SearchMode.SEMANTIC)
        self.assertEqual(response2.context, response.context)

if __name__ == "__main__":
    unittest.main()
