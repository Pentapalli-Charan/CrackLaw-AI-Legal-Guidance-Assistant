import os
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api.main import create_app

class TestAPIService(unittest.TestCase):
    """Integration test suite executing requests against the FastAPI backend REST endpoints."""

    def setUp(self):
        # Instantiate FastAPI app and test client
        self.app = create_app()
        self.client = TestClient(self.app)
        
        # Clear/Set test environment keys
        if "CRACKLAW_API_KEY" in os.environ:
            self.old_key = os.environ["CRACKLAW_API_KEY"]
            del os.environ["CRACKLAW_API_KEY"]
        else:
            self.old_key = None

    def tearDown(self):
        # Restore environment
        if self.old_key:
            os.environ["CRACKLAW_API_KEY"] = self.old_key

    def test_health_check_endpoints(self):
        """Verifies health diagnostics, memory states, and hardware metrics."""
        # 1. Health check
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})

        # 2. Metrics check
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("average_latency_ms", data)

        # 3. Status diagnostics check
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        status_data = response.json()
        self.assertEqual(status_data["status"], "ready")
        self.assertIn("database", status_data)
        self.assertIn("resources", status_data)

    def test_unauthorized_api_key_check(self):
        """Asserts that requests are blocked with HTTP 401 when a key is expected but missing."""
        os.environ["CRACKLAW_API_KEY"] = "secure-test-token-key"
        
        # Request search without header
        response = self.client.post("/api/v1/search", json={"query": "test query"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"], "SecurityError")

    def test_authorized_api_key_check(self):
        """Asserts that requests pass successfully when a valid API key header is present."""
        os.environ["CRACKLAW_API_KEY"] = "secure-test-token-key"
        headers = {"X-API-Key": "secure-test-token-key"}
        
        # Mock retrieval service calls to avoid deep backend invocation
        with patch("src.services.retrieval_service.RetrievalServiceWrapper.search") as mock_search:
            mock_search.return_value = {
                "query": "constitutional rights",
                "results": [],
                "context": "",
                "latency_ms": 1.2
            }
            
            response = self.client.post(
                "/api/v1/search",
                json={"query": "constitutional rights", "top_k": 3},
                headers=headers
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "success")

    def test_chat_generation_endpoint(self):
        """Verifies synchronous chat answers and citations structure formatting."""
        payload = {
            "session_id": "test_session_123",
            "query": "What is the penalty for contract breach?",
            "options": {"temperature": 0.1}
        }
        
        with patch("src.services.chat_service.ChatService.send_message") as mock_send:
            mock_send.return_value = {
                "response_text": "Under Indian Contract Act, Section 73 governs breach.",
                "intent": "Legal Question",
                "rewritten_query": "What is the penalty for contract breach?",
                "citations": [{"text": "Section 73", "source": "Contract Act", "score": 0.9}],
                "confidence_score": 0.85,
                "validation": {"is_valid": True},
                "tokens_used": 150,
                "latency_ms": 32.5,
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "structured_data": {"summary": "Section 73 governs breach."}
            }

            response = self.client.post("/api/v1/chat", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["response_text"], "Under Indian Contract Act, Section 73 governs breach.")
            self.assertEqual(data["confidence_score"], 0.85)
            self.assertEqual(data["citations"][0]["text"], "Section 73")

    def test_chat_streaming_sse_endpoint(self):
        """Verifies real-time SSE stream token outputs."""
        payload = {
            "session_id": "test_stream_session",
            "query": "Hello AI",
        }

        def mock_generator(*args, **kwargs):
            yield {"event": "token", "data": {"token": "Hello"}}
            yield {"event": "token", "data": {"token": " world!"}}
            yield {"event": "done", "data": {"response_text": "Hello world!"}}

        with patch("src.services.chat_service.ChatService.stream_message", side_effect=mock_generator):
            response = self.client.post("/api/v1/chat/stream", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
            
            # Read SSE stream lines
            lines = [line.decode("utf-8") for line in response.iter_bytes() if line]
            full_content = "".join(lines)
            
            self.assertIn("event: token", full_content)
            self.assertIn("data: {\"token\": \"Hello\"}", full_content)
            self.assertIn("event: done", full_content)
            self.assertIn("Hello world!", full_content)

    def test_search_routing_endpoints(self):
        """Verifies routing of general search, semantic search, and hybrid search queries."""
        payload = {"query": "arbitration clauses", "top_k": 2}
        
        with patch("src.services.retrieval_service.RetrievalServiceWrapper.search") as mock_search:
            mock_search.return_value = {
                "query": "arbitration clauses",
                "results": [{"chunk_id": "c1", "document_id": "d1", "text": "snippet", "score": 0.9, "metadata": {}}],
                "context": "snippet",
                "latency_ms": 10.0
            }
            
            # 1. Base search
            response = self.client.post("/api/v1/search", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"][0]["chunk_id"], "c1")
            
            # 2. Semantic search
            response = self.client.post("/api/v1/search/semantic", json=payload)
            self.assertEqual(response.status_code, 200)
            
            # 3. Hybrid search
            response = self.client.post("/api/v1/search/hybrid", json=payload)
            self.assertEqual(response.status_code, 200)

    def test_contract_risk_endpoint(self):
        """Verifies contract risk analysis score returns."""
        payload = {"text": "This contract is governed by state laws..."}
        
        with patch("src.services.contract_service.ContractService.analyze_contract") as mock_risk:
            mock_risk.return_value = {
                "risk_score": 0.45,
                "risk_level": "MEDIUM",
                "features": {"liability": 1.0}
            }
            
            response = self.client.post("/api/v1/contracts/analyze", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["risk_score"], 0.45)
            self.assertEqual(data["risk_level"], "MEDIUM")
            self.assertEqual(data["engineered_features"]["liability"], 1.0)

    def test_document_analyze_and_summarize(self):
        """Verifies analysis and summarization of legal document texts."""
        # 1. Analyze text
        response = self.client.post("/api/v1/documents/analyze", json={"text": "Arbitration Section 10 under the Act"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("legal_feature_scores", response.json())
        self.assertIn("extracted_sections", response.json())
        
        # 2. Summarize text
        with patch("src.services.summary_service.SummaryService.summarize_text") as mock_sum:
            mock_sum.return_value = "Brief Summary text"
            response = self.client.post("/api/v1/summary", json={"text": "Long detailed legal text", "max_length": 100})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["summary"], "Brief Summary text")

    def test_document_upload_validation(self):
        """Asserts uploads block unsupported file types and validate file size constraints."""
        # 1. File validation error: unsupported format (.exe)
        files = {"file": ("test.exe", b"binary content stuff", "application/octet-stream")}
        response = self.client.post("/api/v1/documents/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "FileValidationError")

        # 2. File size limit exceeded error (>10MB)
        massive_data = b"a" * (11 * 1024 * 1024)
        files = {"file": ("document.txt", massive_data, "text/plain")}
        response = self.client.post("/api/v1/documents/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("exceeds the maximum limit", response.json()["message"])
