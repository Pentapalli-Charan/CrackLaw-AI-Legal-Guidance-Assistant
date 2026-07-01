class RetrievalError(Exception):
    """Base exception for all retrieval-related errors in CrackLaw."""
    pass

class EmbeddingGenerationError(RetrievalError):
    """Raised when generating query or document embeddings fails."""
    pass

class VectorStoreError(RetrievalError):
    """Raised when interactions with pgvector or database fail."""
    pass

class DatabaseConnectionError(VectorStoreError):
    """Raised when unable to establish connection to PostgreSQL."""
    pass

class RerankingError(RetrievalError):
    """Raised when Cross-Encoder reranking fails."""
    pass

class SearchError(RetrievalError):
    """Raised when semantic, keyword, or hybrid query search execution fails."""
    pass
