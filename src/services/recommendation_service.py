import logging
from typing import List, Optional
from src.models.model_hub import ModelHub
from src.services.exceptions import ServiceError

logger = logging.getLogger("CrackLaw.Services.RecommendationService")

class RecommendationService:
    """Orchestrates case and document recommendations based on user history similarity."""

    def __init__(self, model_hub: Optional[ModelHub] = None):
        self.model_hub = model_hub or ModelHub()

    def get_recommendations(self, user_id: str, user_history: List[str]) -> List[str]:
        """Queries the Model Hub for top case recommendation matches."""
        try:
            logger.info("Fetching recommendations for user_id=%s...", user_id)
            recs = self.model_hub.recommend(user_id, user_history)
            return recs
        except Exception as e:
            logger.error("Error in recommendation service: %s", str(e))
            raise ServiceError(f"Failed to load recommendations: {e}") from e
