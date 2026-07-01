from enum import Enum

class SearchMode(Enum):
    """Supported search methods in CrackLaw."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class EmbeddingModelType(Enum):
    """Pre-configured legal and general-purpose embedding model architectures."""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    BGE = "bge"
    E5 = "e5"
    MPNET = "mpnet"
    LEGALBERT = "legalbert"
