import unittest
import os
import tempfile
import json
import shutil
import logging
from unittest.mock import MagicMock, patch

from src.config import Config
from src.retrieval.search_result import SearchResultItem, RetrievalResponse
from src.retrieval.retrieval_service import RetrievalService

from src.ai.exceptions import (
    CrackLawAIError,
    LLMProviderError,
    ContextError,
    ValidationError
)
from src.ai.ai_models import Message, Session, AIRequest, AIResponse
from src.ai.token_manager import TokenManager
from src.ai.prompt_templates import TEMPLATE_MAP
from src.ai.prompt_engine import PromptEngine
from src.ai.context_injector import ContextInjector
from src.ai.context_optimizer import ContextOptimizer
from src.ai.intent_detector import IntentDetector, RegexIntentDetector, LLMIntentDetector
from src.ai.query_rewriter import QueryRewriter
from src.ai.llm_gateway import LLMGateway
from src.ai.provider_factory import ProviderFactory
from src.ai.conversation_memory import ConversationMemory
from src.ai.session_manager import SessionManager
from src.ai.citation_generator import CitationGenerator
from src.ai.response_validator import ResponseValidator
from src.ai.hallucination_detector import HallucinationDetector
from src.ai.confidence_engine import ConfidenceEngine
from src.ai.disclaimer_engine import DisclaimerEngine
from src.ai.response_formatter import ResponseFormatter
from src.ai.ai_service import LegalAIService


class TestLegalAIEngine(unittest.TestCase):

    def setUp(self):
        Config._instance = None  # Reset singleton
        self.temp_dir = tempfile.mkdtemp()
        
        # Build sandboxed configurations
        self.config_data = {
            "paths": {
                "datasets_dir": self.temp_dir,
                "raw_dir": os.path.join(self.temp_dir, "raw"),
                "processed_dir": os.path.join(self.temp_dir, "processed"),
                "cleaned_dir": os.path.join(self.temp_dir, "cleaned"),
                "chunks_dir": os.path.join(self.temp_dir, "chunks"),
                "embeddings_dir": os.path.join(self.temp_dir, "embeddings"),
                "metadata_dir": os.path.join(self.temp_dir, "metadata"),
                "cache_dir": os.path.join(self.temp_dir, "cache"),
                "downloads_dir": os.path.join(self.temp_dir, "downloads"),
                "logs_dir": os.path.join(self.temp_dir, "logs")
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(self.temp_dir, "logs", "test_ai.log")
            },
            "retrieval": {
                "top_k": 3,
                "similarity_threshold": 0.25
            },
            "ai": {
                "llm_provider": "gemini",
                "model_name": "gemini-1.5-flash",
                "max_memory_tokens": 1000,
                "max_retries": 2
            }
        }
        
        self.config_path = os.path.join(self.temp_dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f)
            
        self.config = Config(self.config_path)
        self.token_manager = TokenManager()

    def tearDown(self):
        # Shutdown logging to release open files
        logging.shutdown()
        for handler in list(logging.root.handlers):
            try:
                handler.close()
                logging.root.removeHandler(handler)
            except Exception:
                pass
        
        # Attempt to clean up temp dir safely
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        Config._instance = None

    def test_token_manager(self):
        """Tests token estimation and truncation functionality."""
        tm = TokenManager()
        text = "This is a simple sentence to test token estimation."
        tokens = tm.count_tokens(text)
        self.assertTrue(tokens > 5)
        
        # Test Truncation
        truncated = tm.truncate_text(text, 5)
        truncated_tokens = tm.count_tokens(truncated)
        self.assertTrue(truncated_tokens <= 5)
        
        # Test message counting
        messages = [
            Message(role="system", content="System instruction"),
            Message(role="user", content="User question")
        ]
        msg_tokens = tm.count_message_tokens(messages)
        self.assertTrue(msg_tokens > tm.count_tokens("System instruction") + tm.count_tokens("User question"))

    def test_prompt_engine(self):
        """Tests system and user prompt generation from templates."""
        pe = PromptEngine()
        sys_prompt = pe.build_system_prompt("Retrieved Legal Context Text")
        self.assertIn("Retrieved Legal Context Text", sys_prompt)
        self.assertIn("CrackLaw AI", sys_prompt)

        user_prompt = pe.build_user_prompt("Review this NDA.", "Contract Review")
        self.assertIn("Review this NDA.", user_prompt)
        self.assertIn("contract analysis", user_prompt)

    def test_context_injector(self):
        """Tests context format layout wrapping SearchResultItems."""
        ci = ContextInjector()
        items = [
            SearchResultItem(
                chunk_id="chunk_1",
                document_id="doc_a",
                text="The environmental penalty clause.",
                metadata={"act": "Environmental Act", "section": "12", "source": "official_pdf"},
                score=0.9,
                citation="Environmental Act, Sec 12"
            )
        ]
        context = ci.inject_context(items)
        self.assertIn("<document index=\"1\" id=\"chunk_1\">", context)
        self.assertIn("Act: Environmental Act", context)
        self.assertIn("The environmental penalty clause.", context)

    def test_context_optimizer(self):
        """Tests context deduplication, filtering, and budget clipping."""
        co = ContextOptimizer(self.token_manager)
        items = [
            SearchResultItem(chunk_id="ch1", document_id="d1", text="Text 1", score=0.9),
            SearchResultItem(chunk_id="ch2", document_id="d1", text="Text 2", score=0.8),
            SearchResultItem(chunk_id="ch1", document_id="d1", text="Text 1 Duplicate", score=0.7)  # Duplicate id
        ]
        
        # 1. Deduplication
        deduped = co.optimize_context(items, token_budget=1000)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].chunk_id, "ch1")
        self.assertEqual(deduped[1].chunk_id, "ch2")

        # 2. Similarity filter
        filtered = co.optimize_context(items, token_budget=1000, min_similarity=0.85)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, "ch1")

        # 3. Budget Truncation
        # Allocate tiny token budget
        truncated = co.optimize_context(items, token_budget=50)
        self.assertTrue(len(truncated) <= 2)

    def test_regex_intent_detector(self):
        """Tests quick regex pattern matching for legal query intents."""
        rid = RegexIntentDetector()
        self.assertEqual(rid.detect_intent("Please review my agreement clause."), "Contract Review")
        self.assertEqual(rid.detect_intent("Summarize judgment of court."), "Judgment Summary")
        self.assertEqual(rid.detect_intent("find cases regarding copyright."), "Case Search")
        self.assertEqual(rid.detect_intent("hello there"), "General Conversation")
        self.assertIsNone(rid.detect_intent("What is the constitution?"))

    def test_query_rewriter(self):
        """Tests conversational query expansion and resolution of co-references."""
        mock_gateway = MagicMock()
        mock_gateway.generate.return_value = "Judgments on Section 138 of the Negotiable Instruments Act"
        pe = PromptEngine()
        
        qr = QueryRewriter(mock_gateway, pe)
        history = [
            Message(role="user", content="Tell me about section 138 of Negotiable Instruments Act."),
            Message(role="assistant", content="It specifies the penalty for dishonour of cheque.")
        ]
        
        rewritten = qr.rewrite_query("Give me recent judgments on it.", history)
        self.assertEqual(rewritten, "Judgments on Section 138 of the Negotiable Instruments Act")
        mock_gateway.generate.assert_called_once()

    def test_conversation_memory(self):
        """Tests rolling chat message accumulation and budget-aware trimming."""
        # Setup memory with small token allowance
        mem = ConversationMemory(self.token_manager, max_history_tokens=60)
        mem.add_message("system", "Keep this system rule always.")
        mem.add_message("user", "Hello world sentence one.")
        mem.add_message("assistant", "Hi there back response sentence two.")
        
        self.assertEqual(len(mem.get_messages()), 3)
        self.assertEqual(mem.get_messages()[0].role, "system")

        # Add a moderately large message to trigger sliding eviction of older messages
        large_content = "Word " * 30
        mem.add_message("user", large_content)
        
        # System prompt should be preserved, but first user/assistant messages evicted
        messages = mem.get_messages()
        self.assertTrue(len(messages) < 4)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[-1].content, large_content)

    def test_session_manager(self):
        """Tests session storage container operations and backup persistence."""
        sm = SessionManager(self.token_manager, max_history_tokens=500)
        session_id = "test_sess_123"
        
        # Retrieve / Create
        sess = sm.get_session(session_id)
        self.assertIsNotNone(sess["memory"])
        sess["metadata"]["user_name"] = "Alice"
        
        sess["memory"].add_message("user", "Question")
        
        # Persistence Backup & Restore
        backup_file = os.path.join(self.temp_dir, "sessions_backup.json")
        sm.export_sessions_to_json(backup_file)
        self.assertTrue(os.path.exists(backup_file))
        
        # Create new manager to restore state
        sm_new = SessionManager(self.token_manager, max_history_tokens=500)
        sm_new.import_sessions_from_json(backup_file)
        
        restored = sm_new.get_session(session_id)
        self.assertEqual(restored["metadata"]["user_name"], "Alice")
        self.assertEqual(restored["memory"].get_messages()[0].content, "Question")

    def test_citation_generator(self):
        """Tests explicit and implicit extraction of citation targets."""
        cg = CitationGenerator()
        results = [
            SearchResultItem(
                chunk_id="chunk_x",
                document_id="doc_1",
                text="Some text about taxation rules.",
                metadata={"act": "Income Tax Act", "section": "80C", "source": "IT_Act_2026.pdf"},
                score=0.9
            ),
            SearchResultItem(
                chunk_id="chunk_y",
                document_id="doc_2",
                text="Guidance on Environmental clearances.",
                metadata={"act": "Air Act", "section": "21", "judgment": "M.C. Mehta vs Union"},
                score=0.8
            )
        ]

        # 1. Explicit citation via document brackets [2]
        response_text_1 = "Under the guidelines, approvals are required [2]."
        citations_1 = cg.generate_citations(results, response_text_1)
        self.assertEqual(len(citations_1), 1)
        self.assertEqual(citations_1[0]["chunk_id"], "chunk_y")
        self.assertEqual(citations_1[0]["act_name"], "Air Act")
        self.assertEqual(citations_1[0]["reference_type"], "explicit")

        # 2. Implicit citation via lexical keywords (Act name, section name)
        response_text_2 = "Tax benefits are covered under Section 80C of the Income Tax Act."
        citations_2 = cg.generate_citations(results, response_text_2)
        self.assertEqual(len(citations_2), 1)
        self.assertEqual(citations_2[0]["chunk_id"], "chunk_x")
        self.assertEqual(citations_2[0]["reference_type"], "implicit")

    def test_response_validator(self):
        """Tests validation checks on response contents and citations."""
        rv = ResponseValidator()
        
        # Valid
        res_valid = rv.validate("Standard response [1].", [{"chunk_id": "c1"}], [SearchResultItem(chunk_id="c1", document_id="d", text="T")])
        self.assertTrue(res_valid["is_valid"])
        self.assertEqual(res_valid["status"], "PASSED")

        # Empty error
        res_empty = rv.validate("   ", [], [])
        self.assertFalse(res_empty["is_valid"])
        self.assertIn("Empty response", res_empty["errors"][0])

        # Formatting error (unclosed code block)
        res_format = rv.validate("```python\nprint('hello')", [], [])
        self.assertFalse(res_format["is_valid"])
        self.assertIn("Unclosed markdown", res_format["errors"][0])

        # Out of bounds citation index error
        res_idx = rv.validate("References index [5].", [], [SearchResultItem(chunk_id="c1", document_id="d", text="T")])
        self.assertFalse(res_idx["is_valid"])
        self.assertIn("Invalid reference indices", res_idx["errors"][0])

    def test_hallucination_detector(self):
        """Tests lexical overlap overlap checks and flags unsupported claims."""
        hd = HallucinationDetector(lexical_threshold=0.20)
        context = "The environmental protection act mandates penalties up to 5 lakhs under Section 15."
        
        # Well grounded response
        response_grounded = "Penalties of 5 lakhs are issued under Section 15 of the environmental protection act."
        res_g = hd.detect_hallucinations(response_grounded, context)
        self.assertFalse(res_g["hallucination_detected"])
        self.assertEqual(res_g["hallucination_score"], 0.0)

        # Hallucinated response
        response_hallucinated = "The moon is made of green cheese and rabbits live in craters."
        res_h = hd.detect_hallucinations(response_hallucinated, context)
        self.assertTrue(res_h["hallucination_detected"])
        self.assertTrue(res_h["hallucination_score"] > 0.0)

    def test_confidence_engine(self):
        """Tests composite scoring weights of the ConfidenceEngine."""
        ce = ConfidenceEngine()
        results = [SearchResultItem(chunk_id="c1", document_id="d", text="T", score=0.8)]
        citations = [{"chunk_id": "c1"}]
        validation = {"is_valid": True, "errors": [], "warnings": []}
        hallucination = {"hallucination_score": 0.0}

        score = ce.calculate_confidence(results, citations, validation, hallucination)
        self.assertTrue(score >= 0.8)

    def test_disclaimer_engine(self):
        """Tests fetching appropriate templates based on intent."""
        de = DisclaimerEngine()
        self.assertIn("AI-powered legal guidance", de.get_disclaimer("Legal Information"))
        self.assertIn("contract analysis", de.get_disclaimer("Contract Review"))
        self.assertIn("judgment summary", de.get_disclaimer("Judgment Summary"))

    def test_response_formatter(self):
        """Tests extraction of structured parts from raw markdown content."""
        rf = ResponseFormatter()
        raw_text = (
            "SUMMARY: Executive summary of the case.\n"
            "RELEVANT ACTS:\n- Taxation Act 2026\n"
            "RELEVANT SECTIONS:\n- Section 44\n"
            "SUPPORTING CITATIONS:\n- Citation A\n"
            "KEY POINTS:\n- Point 1\n- Point 2\n"
            "SUGGESTED NEXT STEPS:\n- Step 1\n"
        )
        citations = [{"formatted_citation": "Taxation Act, Sec 44"}]
        disclaimer = "Standard Disclaimer"
        
        formatted = rf.format_response(raw_text, citations, 0.9, disclaimer)
        self.assertEqual(formatted["summary"], "Executive summary of the case.")
        self.assertEqual(formatted["relevant_acts"], ["Taxation Act 2026"])
        self.assertEqual(formatted["relevant_sections"], ["Section 44"])
        self.assertIn("Citation A", formatted["supporting_citations"])
        self.assertIn("Taxation Act, Sec 44", formatted["supporting_citations"])
        self.assertEqual(formatted["key_points"], ["Point 1", "Point 2"])
        self.assertEqual(formatted["suggested_next_steps"], ["Step 1"])
        self.assertEqual(formatted["disclaimer"], disclaimer)
        self.assertEqual(formatted["confidence_score"], 0.9)

    @patch("src.ai.llm_gateway.LLMGateway.generate")
    def test_e2e_legal_ai_service(self, mock_llm_generate):
        """Simulates end-to-end reasoning service execution with mocks."""
        # 1. Setup mock LLM answers
        mock_llm_generate.side_effect = [
            "Contract Review", # 1st call: intent detection
            "SUMMARY: Contract Review details.\nRELEVANT ACTS:\n- Contract Act\nRELEVANT SECTIONS:\n- Sec 10\nSUPPORTING CITATIONS:\n- Citation X\nKEY POINTS:\n- P1\nSUGGESTED NEXT STEPS:\n- S1\n" # 2nd call: guidance prompt
        ]

        # 2. Mock Retrieval Engine
        mock_retrieval = MagicMock(spec=RetrievalService)
        mock_retrieval.retrieve.return_value = RetrievalResponse(
            query="Review the contract liability clause.",
            results=[
                SearchResultItem(
                    chunk_id="chunk_id_123",
                    document_id="doc_xyz",
                    text="Agreement contract section 10 mandates consideration.",
                    metadata={"act": "Contract Act", "section": "10"},
                    score=0.9,
                    citation="Contract Act, Sec 10"
                )
            ]
        )

        service = LegalAIService(
            config=self.config,
            retrieval_service=mock_retrieval
        )

        # 3. Trigger generate_response
        response = service.generate_response(
            query="Review the contract liability clause.",
            session_id="session_alice_99",
            options={"llm_provider": "gemini", "model_name": "gemini-1.5-flash"}
        )

        # 4. Verify results
        self.assertIsInstance(response, AIResponse)
        self.assertEqual(response.intent, "Contract Review")
        self.assertEqual(response.structured_data["summary"], "Contract Review details.")
        self.assertIn("Contract Act", response.structured_data["relevant_acts"])
        self.assertIn("Sec 10", response.structured_data["relevant_sections"])
        self.assertTrue(response.confidence_score > 0.5)
        self.assertTrue(response.validation_result["is_valid"])
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0]["chunk_id"], "chunk_id_123")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    unittest.main()
