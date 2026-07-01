import logging
from typing import Dict, Any, Optional
from src.models.model_hub import ModelHub
from src.services.exceptions import ServiceError

logger = logging.getLogger("CrackLaw.Services.ContractService")

class ContractService:
    """Manages legal contract analyses using machine learning risk evaluation classifiers."""

    def __init__(self, model_hub: Optional[ModelHub] = None):
        self.model_hub = model_hub or ModelHub()

    def analyze_contract(self, text: str) -> Dict[str, Any]:
        """Runs the contract text through the Model Hub risk assessment pipeline."""
        if not text.strip():
            return {
                "risk_score": 0.0,
                "risk_level": "LOW",
                "message": "Empty contract text provided."
            }

        try:
            logger.info("Triggering ModelHub contract risk evaluation...")
            result = self.model_hub.calculate_risk(text)
            return result
        except Exception as e:
            logger.error("Error in contract analysis service: %s", str(e))
            raise ServiceError(f"Contract analysis failed: {e}") from e
