from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class SearchResultItem:
    """Represents a single matching document chunk with text and scores."""
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    citation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Converts item instance to a serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "citation": self.citation
        }

@dataclass
class RetrievalResponse:
    """Output wrapper returned by the retrieval engine service containing metadata context."""
    query: str
    results: List[SearchResultItem] = field(default_factory=list)
    context: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts response instance to a serializable dictionary."""
        return {
            "query": self.query,
            "results": [item.to_dict() for item in self.results],
            "context": self.context,
            "latency_ms": self.latency_ms
        }
