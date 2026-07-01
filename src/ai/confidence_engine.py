import logging
from typing import List, Dict, Any
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.AI.ConfidenceEngine")

class ConfidenceEngine:
    """Computes a structured confidence score representing the accuracy and grounding of legal responses."""

    def __init__(self):
        pass

    def calculate_confidence(
        self,
        retrieved_results: List[SearchResultItem],
        citations: List[Dict[str, Any]],
        validation_result: Dict[str, Any],
        hallucination_result: Dict[str, Any]
    ) -> float:
        """Calculates a score between 0.0 and 1.0 based on retrieval, citations, validation, and grounding."""
        
        # 1. Retrieval Similarity Component (weight = 0.3)
        # Average top-k similarity score from retrieval
        if retrieved_results:
            avg_score = sum(getattr(item, "score", 0.0) for item in retrieved_results) / len(retrieved_results)
            # Map average score (usually 0.0 - 1.0 cosine range)
            retrieval_factor = min(1.0, max(0.0, avg_score))
        else:
            retrieval_factor = 0.0

        # 2. Citation Coverage Component (weight = 0.3)
        # Percentage of retrieved chunks linked to citations in the response
        if retrieved_results:
            cited_chunk_ids = {c.get("chunk_id") for c in citations if c.get("chunk_id")}
            retrieved_chunk_ids = {item.chunk_id for item in retrieved_results}
            covered_chunks = retrieved_chunk_ids.intersection(cited_chunk_ids)
            
            # If at least 60% of chunks are cited, reward with high citation factor
            coverage = len(covered_chunks) / len(retrieved_chunk_ids)
            citation_factor = min(1.0, coverage * 1.5)
        else:
            citation_factor = 0.0

        # 3. Grounding Component (weight = 0.3)
        # Compliment of hallucination score
        hallucination_score = hallucination_result.get("hallucination_score", 0.0)
        grounding_factor = max(0.0, 1.0 - hallucination_score)

        # 4. Validation Component (weight = 0.1)
        # Penalyze validation errors or warning counts
        is_valid = validation_result.get("is_valid", True)
        errors = validation_result.get("errors", [])
        warnings = validation_result.get("warnings", [])

        if not is_valid or errors:
            validation_factor = 0.0
        elif warnings:
            validation_factor = 0.5
        else:
            validation_factor = 1.0

        # Combine components
        raw_confidence = (
            (retrieval_factor * 0.3) +
            (citation_factor * 0.3) +
            (grounding_factor * 0.3) +
            (validation_factor * 0.1)
        )

        confidence_score = round(min(1.0, max(0.0, raw_confidence)), 2)
        logger.info(
            "Confidence computed: %.2f (Retrieval: %.2f, Citations: %.2f, Grounding: %.2f, Validation: %.2f)",
            confidence_score, retrieval_factor, citation_factor, grounding_factor, validation_factor
        )

        return confidence_score
