import re
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from src.config import Config

logger = logging.getLogger("CrackLaw.Classification")

class BaseClassifier(ABC):
    """Abstract Base Class for all legal document classifiers."""
    
    @abstractmethod
    def classify(self, text: str, filename: str) -> str:
        """Classifies a document based on text content and metadata.
        
        Args:
            text: The text content of the document.
            filename: The original filename of the document.
            
        Returns:
            A string representing the category (laws, judgments, contracts, etc.).
        """
        pass


class RuleBasedClassifier(BaseClassifier):
    """Rule-based classifier that maps keyphrases and metadata to legal categories."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        ka_config = self.config.get("knowledge_acquisition", {})
        self.category_keywords: Dict[str, List[str]] = ka_config.get("classification", {
            "laws": ["act", "amendment", "statute", "constitution", "repeal"],
            "judgments": ["versus", "vs.", "appeal", "respondent", "petitioner", "bench", "judgment"],
            "contracts": ["agreement", "contract", "lease", "covenant", "warranty", "nda", "parties"],
            "legal_qa": ["question:", "answer:", "q:", "a:", "faq", "query"],
            "legal_nlp": ["token", "annotations", "ner", "tagger", "corpus", "dataset"],
            "notifications": ["notification", "gazette", "circular", "public notice", "issued by"],
            "regulations": ["regulation", "guidelines", "compliance", "directive", "standards"]
        })

    def classify(self, text: str, filename: str) -> str:
        """Classifies the text content and filename using keyword scores."""
        filename_lower = filename.lower()
        text_lower = text.lower() if text else ""
        
        # 1. Simple heuristic overrides based on filename patterns
        if "qa" in filename_lower or "question_answer" in filename_lower:
            return "legal_qa"
        if "contract" in filename_lower or "agreement" in filename_lower or "nda" in filename_lower:
            return "contracts"
        if "judgment" in filename_lower or "appeal" in filename_lower:
            return "judgments"
        if "act" in filename_lower or "law" in filename_lower:
            return "laws"
        if "notification" in filename_lower or "gazette" in filename_lower:
            return "notifications"
        if "regulation" in filename_lower:
            return "regulations"
        if "nlp" in filename_lower or "dataset" in filename_lower or "corpus" in filename_lower:
            return "legal_nlp"

        # 2. Keyphrase frequency scoring
        scores: Dict[str, int] = {cat: 0 for cat in self.category_keywords.keys()}
        
        for category, keywords in self.category_keywords.items():
            # Check keywords in filename (adds substantial weight)
            for kw in keywords:
                if kw in filename_lower:
                    scores[category] += 15
                    
            # Check keywords in document text (frequency-based weight)
            if text_lower:
                for kw in keywords:
                    # Match word boundaries to prevent substring clashes (e.g., 'act' matching 'transaction')
                    matches = re.findall(r"\b" + re.escape(kw) + r"\b", text_lower)
                    scores[category] += len(matches)

        logger.debug("Classification scores for file %s: %s", filename, scores)

        # Get category with highest score
        max_category = "unknown"
        max_score = 0
        
        for category, score in scores.items():
            if score > max_score:
                max_score = score
                max_category = category
                
        # If no keywords matched, classify as unknown
        if max_score == 0:
            return "unknown"
            
        logger.info("Classified '%s' as category: '%s' (Score: %d)", filename, max_category, max_score)
        return max_category
