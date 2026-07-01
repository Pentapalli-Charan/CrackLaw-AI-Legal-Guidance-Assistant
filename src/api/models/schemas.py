from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender, typically 'user' or 'assistant'.")
    content: str = Field(..., description="Text content of the message.")
    timestamp: Optional[float] = Field(None, description="ISO timestamp of the message.")


class Citation(BaseModel):
    text: str = Field(..., description="The matching snippet text.")
    source: Optional[str] = Field(None, description="The origin source name.")
    document_id: Optional[str] = Field(None, description="Associated document register ID.")
    score: float = Field(0.0, description="Similarity matching score.")
    act: Optional[str] = Field(None, description="Matched legislative Act name.")
    section: Optional[str] = Field(None, description="Matched section number.")


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    act: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    source: Optional[str] = None


class LegalFeatureScores(BaseModel):
    char_length: float = 0.0
    word_count: float = 0.0
    count_shall: float = 0.0
    count_indemnify: float = 0.0
    count_liability: float = 0.0
    count_warrants: float = 0.0
    count_breach: float = 0.0
    count_damages: float = 0.0
    count_governing_law: float = 0.0


class DatabaseStats(BaseModel):
    registered_documents: int


class ModelCacheStats(BaseModel):
    loaded_in_memory_cache: List[str]
    cache_size: int


class ResourceStats(BaseModel):
    cpu_usage_percent: float
    ram_usage_percent: float
    ram_available_mb: float
