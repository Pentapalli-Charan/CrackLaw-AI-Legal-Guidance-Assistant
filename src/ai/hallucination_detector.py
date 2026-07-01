import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("CrackLaw.AI.HallucinationDetector")

class HallucinationDetector:
    """Detects hallucinations and unsupported claims in generated legal answers."""

    def __init__(self, lexical_threshold: float = 0.20):
        self.lexical_threshold = lexical_threshold
        # Standard english/legal stop words to filter for overlap checks
        self.stop_words = {
            "a", "an", "the", "and", "but", "or", "in", "on", "at", "to", "for",
            "with", "is", "was", "are", "were", "of", "it", "this", "that", "these",
            "those", "by", "from", "as", "be", "been", "have", "has", "had", "shall",
            "must", "should", "would", "could", "will", "i", "we", "you", "they",
            "he", "she", "it", "not", "no", "any", "some", "such", "there", "their"
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits response text into clean individual sentences using punctuation boundaries."""
        # Simple regex split on sentence endings (.!?) followed by space or newline
        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned_sentences = []
        for s in sentences:
            s_clean = s.strip()
            # Ignore markdown headers and extremely short phrases
            if s_clean and not s_clean.startswith("#") and len(s_clean.split()) >= 4:
                cleaned_sentences.append(s_clean)
        return cleaned_sentences

    def _calculate_overlap(self, sentence: str, context: str) -> float:
        """Calculates token-level overlap ratio between a sentence and the full retrieval context."""
        # Extract alphanumeric words and lowercase them
        sent_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", sentence.lower()))
        context_words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", context.lower()))

        # Filter out common stop words to focus on substantive nouns/verbs/terms
        meaningful_sent_words = sent_words - self.stop_words
        if not meaningful_sent_words:
            return 1.0  # Empty or stopword-only sentence is trivial

        matching_words = meaningful_sent_words.intersection(context_words)
        return len(matching_words) / len(meaningful_sent_words)

    def detect_hallucinations(
        self,
        response_text: str,
        retrieved_context: str,
        llm_gateway: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Evaluates grounding. Combines lexical overlap scoring and optional LLM NLI fact verification."""
        unsupported_claims = []
        sentences = self._split_into_sentences(response_text)
        
        if not sentences or not retrieved_context.strip():
            # If there's no context, everything generated might be unsupported unless it is general convo
            return {
                "hallucination_detected": len(sentences) > 0,
                "hallucination_score": 1.0 if len(sentences) > 0 else 0.0,
                "unsupported_claims": ["No retrieval context provided to ground the response."] if sentences else [],
                "sentence_scores": []
            }

        sentence_scores = []
        for sent in sentences:
            overlap = self._calculate_overlap(sent, retrieved_context)
            sentence_scores.append({"sentence": sent, "score": overlap})
            
            if overlap < self.lexical_threshold:
                unsupported_claims.append(sent)

        # Calculate a baseline hallucination score (ratio of unsupported sentences)
        hallucination_score = len(unsupported_claims) / len(sentences) if sentences else 0.0
        hallucination_detected = len(unsupported_claims) > 0

        # Optional LLM NLI check (if LLM gateway is supplied)
        # Excellent for complex semantic synthesis not captured by strict word overlap
        if llm_gateway and hallucination_detected:
            try:
                verify_prompt = (
                    f"Check if the following statements are supported by the provided Legal Context.\n"
                    f"Legal Context:\n{retrieved_context}\n\n"
                    f"Statements to verify:\n" + "\n".join([f"- {s}" for s in unsupported_claims]) + "\n\n"
                    f"Instructions: Classify each statement as 'SUPPORTED' or 'UNSUPPORTED' based strictly on the context.\n"
                    f"Format: Statement | Classification | Reason\n"
                    f"Output:"
                )
                
                llm_response = llm_gateway.generate(
                    prompt=verify_prompt,
                    system_prompt="You are a factual verification assistant. Verify if statements are grounded in context.",
                    temperature=0.0,
                    max_tokens=512
                )
                
                # Parse lines, filtering out claims verified as SUPPORTED by the LLM
                refined_claims = []
                for line in llm_response.split("\n"):
                    if "UNSUPPORTED" in line:
                        refined_claims.append(line.strip())
                
                if refined_claims:
                    unsupported_claims = refined_claims
                    hallucination_score = len(unsupported_claims) / len(sentences)
                else:
                    unsupported_claims = []
                    hallucination_score = 0.0
                    hallucination_detected = False
                    
            except Exception as e:
                logger.error("LLM-based NLI hallucination check failed: %s", str(e))

        logger.info(
            "Hallucination detection completed. Detected: %s, Score: %.2f",
            hallucination_detected, hallucination_score
        )

        return {
            "hallucination_detected": hallucination_detected,
            "hallucination_score": round(hallucination_score, 2),
            "unsupported_claims": unsupported_claims,
            "sentence_scores": sentence_scores
        }
