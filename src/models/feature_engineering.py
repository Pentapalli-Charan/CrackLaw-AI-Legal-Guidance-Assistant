import re
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("CrackLaw.Models.FeatureEngineering")

class LegalFeatureExtractor:
    """Extracts structural, lexical, and legal-specific features from raw document texts."""

    def __init__(self):
        # Target keywords indicative of obligations, liability, and risks
        self.keywords = [
            "shall", "must", "agree", "undertake",
            "indemnify", "hold harmless", "liable", "liability",
            "breach", "violation", "default",
            "termination", "expire", "cancel",
            "warranty", "guarantee", "represent",
            "confidential", "proprietary", "nondisclosure",
            "governing law", "jurisdiction", "arbitration"
        ]

    def extract_features_from_text(self, text: str) -> Dict[str, float]:
        """Parses a single text input to extract numerical legal features."""
        if not text:
            return {kw: 0.0 for kw in self.keywords} | {
                "char_length": 0.0,
                "word_count": 0.0,
                "sentence_count": 0.0,
                "avg_word_len": 0.0
            }

        cleaned = text.strip()
        words = cleaned.split()
        char_len = len(cleaned)
        word_cnt = len(words)
        
        # Simple sentence count estimation
        sentences = re.split(r"[.!?]+", cleaned)
        sent_cnt = max(1.0, len([s for s in sentences if s.strip()]))

        # Calculate raw counts of each legal keyword
        features = {}
        for kw in self.keywords:
            # Match word boundary or exact match for multi-word keywords
            pattern = rf"\b{re.escape(kw)}\b"
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            features[f"count_{kw.replace(' ', '_')}"] = float(len(matches))

        # Basic metadata features
        features["char_length"] = float(char_len)
        features["word_count"] = float(word_cnt)
        features["sentence_count"] = float(sent_cnt)
        features["avg_word_len"] = float(char_len / word_cnt) if word_cnt > 0 else 0.0

        return features

    def transform_to_array(self, texts: List[str]) -> np.ndarray:
        """Transforms a batch of texts into a 2D numpy feature matrix."""
        features_list = []
        for text in texts:
            feats = self.extract_features_from_text(text)
            # Sort keys to ensure consistent column ordering
            ordered_vals = [feats[k] for k in sorted(feats.keys())]
            features_list.append(ordered_vals)
        
        return np.array(features_list, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Returns ordered names of features generated in transform_to_array."""
        # Create a dummy run to get keys
        dummy_feats = self.extract_features_from_text("shall indemnity")
        return sorted(dummy_feats.keys())
