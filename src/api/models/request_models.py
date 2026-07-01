from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The message query to send to the assistant.")
    session_id: str = Field(..., min_length=1, description="Chat session/thread unique identifier.")
    options: Optional[Dict[str, Any]] = Field(None, description="Generation overrides for the LLM providers.")


class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata key filters to apply.")
    top_k: int = Field(5, ge=1, le=50, description="Maximum matches to return.")
    min_similarity: float = Field(0.25, ge=0.0, le=1.0, description="Minimum similarity matching threshold.")


class DocumentAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw text document content to analyze.")


class ContractRiskRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Contract text body to calculate risk index score.")


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User unique identifier.")
    user_history: List[str] = Field(..., description="List of document/case IDs the user previously opened.")


class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Legal text to summarize.")
    max_length: int = Field(500, ge=10, le=5000, description="Target character length limit.")
