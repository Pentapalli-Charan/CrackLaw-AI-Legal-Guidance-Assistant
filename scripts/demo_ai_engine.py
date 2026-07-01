import os
import sys
import json
import logging
from unittest.mock import MagicMock

# Adjust path to import src modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.retrieval.search_result import SearchResultItem, RetrievalResponse
from src.retrieval.retrieval_service import RetrievalService
from src.ai.ai_service import LegalAIService
from src.ai.ai_models import AIResponse

def setup_demo_logging():
    """Sets up a clean stdout logging formatter for the demonstration."""
    logger = logging.getLogger("CrackLaw")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def run_demo():
    print("=" * 70)
    print("          CRACKLAW LEGAL AI ENGINE - DEVELOPER DEMONSTRATION")
    print("=" * 70)

    # 1. Setup mock configuration environment
    config = Config()
    
    # 2. Mock Retrieval Service with pre-seeded legal chunks
    mock_retrieval = MagicMock(spec=RetrievalService)
    
    # Pre-seeded chunks for simulated query matches
    simulated_chunks = [
        SearchResultItem(
            chunk_id="chk_contract_101",
            document_id="doc_contract_act",
            text=(
                "Section 73 of the Indian Contract Act, 1872 deals with compensation for loss "
                "or damage caused by breach of contract. When a contract has been broken, the party "
                "who suffers by such breach is entitled to receive, from the party who has broken the "
                "contract, compensation for any loss or damage caused to him thereby, which naturally "
                "arose in the usual course of things."
            ),
            metadata={
                "act": "Indian Contract Act, 1872",
                "section": "73",
                "chapter": "VI",
                "source": "Indian_Contract_Act_1872.pdf"
            },
            score=0.92,
            citation="Indian Contract Act, 1872, Sec 73"
        ),
        SearchResultItem(
            chunk_id="chk_liability_205",
            document_id="doc_liability_rules",
            text=(
                "In commercial agreements, limitation of liability clauses typically cap damages "
                "at the total fees paid under the contract in the preceding 12 months. However, exclusions "
                "apply for gross negligence, willful misconduct, and breach of confidentiality obligations."
            ),
            metadata={
                "act": "Commercial NDA Guidelines",
                "section": "Clause 14",
                "source": "Standard_NDA_Template.pdf"
            },
            score=0.88,
            citation="Commercial NDA Guidelines, Clause 14"
        )
    ]
    mock_retrieval.retrieve.return_value = RetrievalResponse(
        query="NDA liability limitations under Section 73",
        results=simulated_chunks
    )

    # 3. Instantiate the Legal AI Service Facade
    # We will patch the LLM Gateway's actual HTTP calls to return simulated GPT/Claude responses
    service = LegalAIService(
        config=config,
        retrieval_service=mock_retrieval
    )

    # We patch the generate call inside the LLM Gateway to simulate the LLM output
    # 1st query: "Please review my contract NDA liability limits under section 73." (Intent: Contract Review)
    # 2nd query: "hello there, I am Charan. Who are you?" (Intent: General Conversation)
    # 3rd query: "Can you find court rulings on it?" (Intent: Case Search)
    simulated_llm_responses = [
        # Call 1: Intent detection for Query 1
        "Contract Review",
        # Call 2: Core reasoning completion for Query 1
        (
            "SUMMARY: Indian contract law caps damages based on actual loss. Under Section 73 of the Indian Contract Act, 1872 [1], "
            "damages are limited to those arising naturally. Under commercial guidelines, limitation of liability clauses typically cap "
            "damages at fees paid, excluding gross negligence or confidentiality breaches [2].\n\n"
            "RELEVANT ACTS:\n- Indian Contract Act, 1872\n- Commercial NDA Guidelines\n\n"
            "RELEVANT SECTIONS:\n- Section 73\n- Clause 14\n\n"
            "SUPPORTING CITATIONS:\n- Indian Contract Act, 1872, Sec 73\n- Commercial NDA Guidelines, Clause 14\n\n"
            "KEY POINTS:\n- Damages must arise naturally from the breach.\n- Standard cap is usually 12-month fees.\n"
            "- Gross negligence and confidentiality breaches are excluded from caps.\n\n"
            "SUGGESTED NEXT STEPS:\n- Review if confidentiality breaches are uncapped in the current agreement.\n"
            "- Draft a mutual limitation clause capped at fees paid."
        ),
        # Call 3: Intent detection for Query 2
        "General Conversation",
        # Call 4: Core reasoning completion for Query 2 (No retrieval needed!)
        "Hello Charan! I am CrackLaw AI, your dedicated legal guidance assistant. How can I assist you with legal research, contract review, or risk analysis today?",
        
        # Call 5: Intent detection for Query 3
        "Case Search",
        # Call 6: Query rewriting for Query 3
        "Find court rulings on Section 73 Indian Contract Act liability limits",
        # Call 7: Core reasoning completion for Query 3
        (
            "SUMMARY: Landmark judgments on Section 73 of the Indian Contract Act limit claims to actual proven losses, rejecting speculative damages [1]. "
            "In Hadley v Baxendale, damages are restricted to natural consequences of breach. In Maula Bux v Union of India, "
            "the Supreme Court held that reasonable compensation is payable but must be proved unless assessment is impossible.\n\n"
            "RELEVANT ACTS:\n- Indian Contract Act, 1872\n\n"
            "RELEVANT SECTIONS:\n- Section 73\n\n"
            "SUPPORTING CITATIONS:\n- Indian Contract Act, 1872, Sec 73\n\n"
            "KEY POINTS:\n- Hadley v Baxendale established the test of remoteness of damages.\n"
            "- Maula Bux v Union of India restricts forfeiture of earnest money without proof of damage.\n\n"
            "SUGGESTED NEXT STEPS:\n- Examine the liquidated damages clause for potential penalties.\n"
            "- Gather proof of actual loss to satisfy the evidentiary standards of Section 73."
        )
    ]

    session_id = "session_demo_charan"

    # Patch the generate method in LLMGateway
    with MagicMock() as mock_generate:
        mock_generate.side_effect = simulated_llm_responses
        service.llm_gateway.generate = mock_generate

        # ----------------------------------------------------
        # TURN 1: Contract Review Query
        # ----------------------------------------------------
        query_1 = "Please review my contract NDA liability limits under section 73."
        print(f"\n>>> USER QUERY 1: '{query_1}'")
        print("Processing reasoning pipeline...")
        
        resp_1 = service.generate_response(query_1, session_id)
        
        print("\n=== SYSTEM PIPELINE LOGS ===")
        print(f"Detected Intent:  {resp_1.intent}")
        print(f"Rewritten Query:  {resp_1.rewritten_query}")
        print(f"Citations Found:  {len(resp_1.citations)}")
        print(f"Confidence Score: {resp_1.confidence_score * 100}%")
        print(f"Latency:          {resp_1.latency_ms:.2f} ms")
        print(f"Tokens Count:     {resp_1.tokens_used}")
        print(f"Validation State: {resp_1.validation_result['status']}")

        print("\n=== STRUCTURED RESPONSE DATA ===")
        print(f"Summary:\n{resp_1.structured_data['summary']}")
        print(f"Relevant Acts: {resp_1.structured_data['relevant_acts']}")
        print(f"Relevant Sections: {resp_1.structured_data['relevant_sections']}")
        print(f"Key Points:")
        for pt in resp_1.structured_data["key_points"]:
            print(f"  - {pt}")
        print(f"Next Steps:")
        for ns in resp_1.structured_data["suggested_next_steps"]:
            print(f"  - {ns}")
        print(f"Disclaimer:\n{resp_1.structured_data['disclaimer']}")
        print("-" * 70)

        # ----------------------------------------------------
        # TURN 2: General Greeting (Retrieval is skipped)
        # ----------------------------------------------------
        query_2 = "hello there, I am Charan. Who are you?"
        print(f"\n>>> USER QUERY 2: '{query_2}'")
        print("Processing reasoning pipeline...")
        
        resp_2 = service.generate_response(query_2, session_id)
        
        print("\n=== SYSTEM PIPELINE LOGS ===")
        print(f"Detected Intent:  {resp_2.intent}")
        print(f"Citations Found:  {len(resp_2.citations)} (Should be 1 fallback context)")
        print(f"Context Text:     '{resp_2.retrieved_context}' (Should be empty for general conversation)")
        print(f"Response Summary: {resp_2.response_text}")
        print("-" * 70)

        # ----------------------------------------------------
        # TURN 3: Follow-up query requiring co-reference resolution
        # ----------------------------------------------------
        query_3 = "Can you find court rulings on it?"
        print(f"\n>>> USER QUERY 3: '{query_3}'")
        print("Processing reasoning pipeline...")
        
        resp_3 = service.generate_response(query_3, session_id)
        
        print("\n=== SYSTEM PIPELINE LOGS ===")
        print(f"Detected Intent:  {resp_3.intent}")
        print(f"Rewritten Query:  {resp_3.rewritten_query} (Resolved 'it' to Section 73 Indian Contract Act liability limits)")
        print(f"Confidence Score: {resp_3.confidence_score * 100}%")
        print(f"Validation State: {resp_3.validation_result['status']}")

        print("\n=== STRUCTURED RESPONSE DATA ===")
        print(f"Summary:\n{resp_3.structured_data['summary']}")
        print(f"Key Points:")
        for pt in resp_3.structured_data["key_points"]:
            print(f"  - {pt}")
        print(f"Disclaimer:\n{resp_3.structured_data['disclaimer']}")
        print("=" * 70)
        print("              DEMONSTRATION RUN COMPLETE - SUCCESS")
        print("=" * 70)

if __name__ == "__main__":
    setup_demo_logging()
    run_demo()
