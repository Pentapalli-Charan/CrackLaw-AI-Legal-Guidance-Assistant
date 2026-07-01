import time
import logging
from typing import Dict, Any, Optional, Generator
from src.ai.ai_service import LegalAIService
from src.services.exceptions import ServiceError

logger = logging.getLogger("CrackLaw.Services.ChatService")

class ChatService:
    """Orchestrates conversations with the Legal AI Engine reasoning pipeline."""

    def __init__(self, ai_service: Optional[LegalAIService] = None):
        self.ai_service = ai_service or LegalAIService()

    def send_message(self, session_id: str, query: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Submits query to AI service and returns the complete reasoning response."""
        try:
            logger.info("Executing chat query for session %s", session_id)
            ai_response = self.ai_service.generate_response(query, session_id, options)
            
            # Serialize citations cleanly
            citations_list = []
            for c in ai_response.citations:
                if hasattr(c, "to_dict"):
                    citations_list.append(c.to_dict())
                elif hasattr(c, "__dict__"):
                    citations_list.append(c.__dict__)
                else:
                    citations_list.append(str(c))

            return {
                "response_text": ai_response.response_text,
                "intent": ai_response.intent,
                "rewritten_query": ai_response.rewritten_query,
                "citations": citations_list,
                "confidence_score": ai_response.confidence_score,
                "validation": ai_response.validation_result,
                "tokens_used": ai_response.tokens_used,
                "latency_ms": ai_response.latency_ms,
                "provider": ai_response.provider,
                "model": ai_response.model,
                "structured_data": ai_response.structured_data
            }
        except Exception as e:
            logger.error("Error in chat service send_message: %s", str(e))
            raise ServiceError(f"Chat generation error: {e}") from e

    def stream_message(self, session_id: str, query: str, options: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """Streams AI responses word-by-word, ending with the full structured result metadata block."""
        try:
            logger.info("Starting chat stream for session %s", session_id)
            response_dict = self.send_message(session_id, query, options)
            text = response_dict["response_text"]
            
            # Stream tokens sequentially
            words = text.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield {
                    "event": "token",
                    "data": {"token": token}
                }
                time.sleep(0.015)  # 15ms artificial lag for smooth reading flow
            
            # Conclude stream yielding final metadata payload
            yield {
                "event": "done",
                "data": response_dict
            }
        except Exception as e:
            logger.error("Error in chat service stream_message: %s", str(e))
            yield {
                "event": "error",
                "data": {"detail": str(e)}
            }
