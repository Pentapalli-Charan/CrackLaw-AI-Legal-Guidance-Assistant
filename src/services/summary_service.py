import logging
from typing import Optional
from src.ai.ai_service import LegalAIService
from src.services.exceptions import ServiceError

logger = logging.getLogger("CrackLaw.Services.SummaryService")

class SummaryService:
    """Orchestrates legal text summarization utilizing the AI Engine's generative models."""

    def __init__(self, ai_service: Optional[LegalAIService] = None):
        self.ai_service = ai_service or LegalAIService()

    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """Invokes the Legal AI Engine to generate a concise summary of the given text."""
        if not text.strip():
            return ""

        try:
            logger.info("Triggering legal text summarization (target length: %d chars)...", max_length)
            prompt = f"Please summarize the following legal text or contract concisely in maximum {max_length} characters:\n\n{text}"
            
            # Temporary session for summarization
            session_id = "summary_session_temp"
            response = self.ai_service.generate_response(prompt, session_id)
            
            return response.response_text
        except Exception as e:
            logger.error("Error in SummaryService: %s", str(e))
            raise ServiceError(f"Text summarization failed: {e}") from e
