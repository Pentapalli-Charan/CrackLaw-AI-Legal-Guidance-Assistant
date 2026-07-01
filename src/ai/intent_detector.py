import re
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

logger = logging.getLogger("CrackLaw.AI.IntentDetector")

class BaseIntentDetector(ABC):
    """Abstract base class for all intent classification sub-modules."""

    @abstractmethod
    def detect_intent(self, query: str) -> Optional[str]:
        """Classifies the intent of a query, returning the intent name or None if uncertain."""
        pass


class RegexIntentDetector(BaseIntentDetector):
    """Fast, regex-based intent classification for common queries."""

    def __init__(self):
        # Compiled patterns mapping keywords to intents
        self.rules = {
            "Contract Review": re.compile(
                r"(?i)\b(review contract|analyze agreement|contract clause|nda review|indemnification clause|liability limit|agreement clause)\b"
            ),
            "Judgment Summary": re.compile(
                r"(?i)\b(summarize judgment|judgment summary|case summary|summarize case|judgment details|ruling summary)\b"
            ),
            "Case Search": re.compile(
                r"(?i)\b(find cases|search cases|case law|precedents|court judgments|court rulings|citations for)\b"
            ),
            "Legal Risk Analysis": re.compile(
                r"(?i)\b(legal risk|compliance risk|liabilities|risk analysis|exposure assessment|risk factor)\b"
            ),
            "Document Analysis": re.compile(
                r"(?i)\b(analyze document|explain notice|legal notice|explain letter|notice analysis|document explanation)\b"
            ),
            "General Conversation": re.compile(
                r"(?i)\b(hello|hi|hey|greetings|how are you|who are you|help|thank you|thanks)\b"
            )
        }

    def detect_intent(self, query: str) -> Optional[str]:
        for intent, pattern in self.rules.items():
            if pattern.search(query):
                logger.debug("Regex pattern matched intent: %s", intent)
                return intent
        return None


class LLMIntentDetector(BaseIntentDetector):
    """Semantic intent classification using the LLM Gateway."""

    def __init__(self, llm_gateway: Any, prompt_engine: Any):
        self.llm_gateway = llm_gateway
        self.prompt_engine = prompt_engine

    def detect_intent(self, query: str) -> Optional[str]:
        prompt = self.prompt_engine.build_intent_detection_prompt(query)
        try:
            # Call low-temperature request for categorical accuracy
            response_text = self.llm_gateway.generate(
                prompt=prompt,
                system_prompt="You are an intent detection helper. Classify the user query exactly.",
                temperature=0.0,
                max_tokens=20
            )
            intent = response_text.strip()
            
            # Map or validate intent
            valid_intents = {
                "Legal Information", "Legal Research", "Contract Review",
                "Document Analysis", "Case Search", "Judgment Summary",
                "Legal Risk Analysis", "General Conversation"
            }
            # Clean up the output in case LLM added extra punctuation/quotes
            intent_clean = intent.replace('"', '').replace("'", "").strip()
            for val in valid_intents:
                if val.lower() == intent_clean.lower():
                    logger.info("LLM classified intent: %s", val)
                    return val

            # Partial match fallback
            for val in valid_intents:
                if val.lower() in intent_clean.lower():
                    logger.info("LLM partial match classified intent: %s", val)
                    return val

            logger.warning("LLM returned unknown intent classification: '%s'", intent)
            return "Legal Information" # Fallback default
        except Exception as e:
            logger.error("LLM intent detection failed: %s", str(e))
            return None


class IntentDetector:
    """Orchestrator managing multiple intent detection systems in a pipeline."""

    def __init__(self, detectors: Optional[List[BaseIntentDetector]] = None):
        self.detectors = detectors or [RegexIntentDetector()]

    def register_detector(self, detector: BaseIntentDetector) -> None:
        self.detectors.append(detector)

    def detect_intent(self, query: str) -> str:
        """Runs the query through the registered intent detectors, falling back to Legal Information."""
        for detector in self.detectors:
            try:
                intent = detector.detect_intent(query)
                if intent:
                    return intent
            except Exception as e:
                logger.error("Intent detector failed execution: %s", str(e))

        # Fallback category if all detectors return None
        return "Legal Information"
