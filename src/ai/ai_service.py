import time
import logging
from typing import Dict, Any, Optional, List

from src.config import Config
from src.retrieval.retrieval_service import RetrievalService
from src.ai.exceptions import CrackLawAIError, LLMProviderError, ContextError
from src.ai.ai_models import AIRequest, AIResponse, Message
from src.ai.token_manager import TokenManager
from src.ai.prompt_engine import PromptEngine
from src.ai.context_injector import ContextInjector
from src.ai.context_optimizer import ContextOptimizer
from src.ai.intent_detector import IntentDetector, RegexIntentDetector, LLMIntentDetector
from src.ai.query_rewriter import QueryRewriter
from src.ai.llm_gateway import LLMGateway
from src.ai.session_manager import SessionManager
from src.ai.citation_generator import CitationGenerator
from src.ai.response_validator import ResponseValidator
from src.ai.hallucination_detector import HallucinationDetector
from src.ai.confidence_engine import ConfidenceEngine
from src.ai.disclaimer_engine import DisclaimerEngine
from src.ai.response_formatter import ResponseFormatter

logger = logging.getLogger("CrackLaw.AI.Service")

class LegalAIService:
    """Main facade class orchestrating the Legal AI Engine reasoning pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        retrieval_service: Optional[RetrievalService] = None,
        token_manager: Optional[TokenManager] = None,
        prompt_engine: Optional[PromptEngine] = None,
        context_injector: Optional[ContextInjector] = None,
        context_optimizer: Optional[ContextOptimizer] = None,
        intent_detector: Optional[IntentDetector] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        llm_gateway: Optional[LLMGateway] = None,
        session_manager: Optional[SessionManager] = None,
        citation_generator: Optional[CitationGenerator] = None,
        response_validator: Optional[ResponseValidator] = None,
        hallucination_detector: Optional[HallucinationDetector] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        disclaimer_engine: Optional[DisclaimerEngine] = None,
        response_formatter: Optional[ResponseFormatter] = None
    ):
        self.config = config or Config()
        
        # Instantiate Retrieval Engine if not supplied
        self.retrieval_service = retrieval_service or RetrievalService(self.config)
        
        # Instantiate helpers
        self.token_manager = token_manager or TokenManager()
        self.prompt_engine = prompt_engine or PromptEngine()
        self.context_injector = context_injector or ContextInjector()
        self.context_optimizer = context_optimizer or ContextOptimizer(self.token_manager, self.context_injector)
        self.llm_gateway = llm_gateway or LLMGateway(self.config)
        
        # Extensible Intent Detection registration
        if intent_detector:
            self.intent_detector = intent_detector
        else:
            self.intent_detector = IntentDetector()
            # Register Regex first (default list inside IntentDetector)
            # Register LLM Intent Detector as fallback
            self.intent_detector.register_detector(
                LLMIntentDetector(self.llm_gateway, self.prompt_engine)
            )

        self.query_rewriter = query_rewriter or QueryRewriter(self.llm_gateway, self.prompt_engine)
        
        # Session memory configurations
        ai_settings = self.config.get("ai", {})
        max_mem_tokens = ai_settings.get("max_memory_tokens", 4096)
        self.session_manager = session_manager or SessionManager(self.token_manager, max_history_tokens=max_mem_tokens)
        
        # Output verification and alignment pipelines
        self.citation_generator = citation_generator or CitationGenerator()
        self.response_validator = response_validator or ResponseValidator()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.confidence_engine = confidence_engine or ConfidenceEngine()
        self.disclaimer_engine = disclaimer_engine or DisclaimerEngine()
        self.response_formatter = response_formatter or ResponseFormatter()

    def generate_response(
        self,
        query: str,
        session_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AIResponse:
        """Executes the complete reasoning and retrieval pipeline for a user query."""
        t_start = time.time()
        options = options or {}
        
        logger.info("New query received in session '%s': '%s'", session_id, query)

        # 1. Retrieve session container (thread-safe)
        session_container = self.session_manager.get_session(session_id)
        memory = session_container["memory"]
        history = memory.get_messages()

        # Get settings merged with option overrides
        ai_settings = self.config.get("ai", {})
        provider = options.get("llm_provider") or ai_settings.get("llm_provider", "gemini")
        model = options.get("model_name") or ai_settings.get("model_name", "gemini-1.5-flash")
        temperature = options.get("temperature", 0.2)
        max_tokens = options.get("max_tokens", 1024)
        top_p = options.get("top_p", 0.95)

        # 2. Intent Detection
        intent = self.intent_detector.detect_intent(query)
        logger.info("Query intent detected: '%s'", intent)

        # 3. Query Rewriting (if there is history to resolve pronouns)
        rewritten_query = query
        if history and intent != "General Conversation":
            rewritten_query = self.query_rewriter.rewrite_query(query, history)

        # 4. Context Retrieval (Skip if general conversational topic)
        retrieved_results = []
        formatted_context = ""
        
        if intent != "General Conversation":
            try:
                ret_settings = self.config.retrieval_settings
                top_k = options.get("top_k") or ret_settings.get("top_k", 5)
                min_sim = options.get("min_similarity") or ret_settings.get("similarity_threshold", 0.25)
                token_budget = options.get("context_token_budget", 2048)

                # Fetch matching document chunks
                logger.info("Triggering retrieval for: '%s'", rewritten_query)
                ret_response = self.retrieval_service.retrieve(
                    query=rewritten_query,
                    top_k=top_k * 2,  # Fetch extra so optimizer has choices to prune
                    min_similarity=min_sim
                )
                
                # Fit chunks into token budget
                optimized_results = self.context_optimizer.optimize_context(
                    results=ret_response.results,
                    token_budget=token_budget,
                    min_similarity=min_sim
                )
                retrieved_results = optimized_results
                
                # Format segments into prompt blocks
                formatted_context = self.context_injector.inject_context(optimized_results)

            except Exception as e:
                logger.error("Retrieval failed during AI response generation: %s", str(e))
                # Graceful recovery: proceed without context and issue warning
                formatted_context = ""
                retrieved_results = []

        # 5. Prompts Assembly
        system_prompt = self.prompt_engine.build_system_prompt(formatted_context)
        user_prompt = self.prompt_engine.build_user_prompt(query, intent)

        # 6. Call LLM Gateway
        try:
            raw_response = self.llm_gateway.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                provider_name=provider,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
        except Exception as e:
            logger.error("LLM Gateway failed: %s", str(e))
            # Return a user-friendly error payload that doesn't crash the application
            latency_ms = (time.time() - t_start) * 1000
            err_disclaimer = self.disclaimer_engine.get_disclaimer("default")
            
            return AIResponse(
                response_text="Error: The AI Engine failed to generate a response. Please check configuration and try again.",
                intent=intent,
                rewritten_query=rewritten_query,
                retrieved_context=formatted_context,
                structured_data={
                    "summary": "AI Engine Generation Failure.",
                    "relevant_acts": [],
                    "relevant_sections": [],
                    "supporting_citations": [],
                    "key_points": [f"Error Details: {str(e)}"],
                    "suggested_next_steps": ["Check network credentials", "Try alternative provider"],
                    "disclaimer": err_disclaimer,
                    "confidence_score": 0.0
                },
                citations=[],
                confidence_score=0.0,
                validation_result={"is_valid": False, "errors": [str(e)], "warnings": []},
                tokens_used=0,
                latency_ms=latency_ms,
                provider=provider,
                model=model
            )

        # 7. Citations extraction and alignment
        citations = self.citation_generator.generate_citations(retrieved_results, raw_response)

        # 8. Validation pipeline
        validation_result = self.response_validator.validate(raw_response, citations, retrieved_results)

        # 9. Hallucination detection NLI/lexical
        hallucination_result = self.hallucination_detector.detect_hallucinations(
            response_text=raw_response,
            retrieved_context=formatted_context,
            llm_gateway=self.llm_gateway if intent != "General Conversation" else None
        )

        # 10. Confidence score generation
        confidence_score = self.confidence_engine.calculate_confidence(
            retrieved_results=retrieved_results,
            citations=citations,
            validation_result=validation_result,
            hallucination_result=hallucination_result
        )

        # 11. Disclaimer generation
        disclaimer = self.disclaimer_engine.get_disclaimer(intent)

        # 12. Final structured formatting
        structured_data = self.response_formatter.format_response(
            raw_response=raw_response,
            citations=citations,
            confidence_score=confidence_score,
            disclaimer=disclaimer
        )

        # 13. Update short-term session conversation history
        # We save user prompt and the final structured summary / answer text
        memory.add_message("user", query)
        memory.add_message("assistant", structured_data["summary"])

        # Calculate metrics
        latency_ms = (time.time() - t_start) * 1000
        prompt_tokens = self.token_manager.count_tokens(system_prompt + user_prompt)
        completion_tokens = self.token_manager.count_tokens(raw_response)
        total_tokens = prompt_tokens + completion_tokens

        logger.info(
            "Response generation completed successfully in %.2f ms. Total tokens used: %d",
            latency_ms, total_tokens
        )

        return AIResponse(
            response_text=raw_response,
            intent=intent,
            rewritten_query=rewritten_query,
            retrieved_context=formatted_context,
            structured_data=structured_data,
            citations=citations,
            confidence_score=confidence_score,
            validation_result=validation_result,
            tokens_used=total_tokens,
            latency_ms=latency_ms,
            provider=provider,
            model=model
        )
