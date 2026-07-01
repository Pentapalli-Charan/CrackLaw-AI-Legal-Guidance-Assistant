from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from src.api.models.schemas import ChatMessage, Citation, SearchResult, DatabaseStats, ModelCacheStats, ResourceStats

class BaseResponse(BaseModel):
    status: str = Field("success", description="Status code indicator, usually 'success' or 'error'.")
    error: Optional[str] = Field(None, description="Detailed error description, if status is error.")
    latency_ms: float = Field(0.0, description="Response processing duration in milliseconds.")


class ChatResponse(BaseResponse):
    response_text: str
    intent: str
    rewritten_query: str
    citations: List[Citation]
    confidence_score: float
    provider: str
    model: str
    structured_data: Dict[str, Any]


class SearchResponse(BaseResponse):
    query: str
    results: List[SearchResult]
    context: str


class DocumentUploadResponse(BaseResponse):
    document_id: str
    filename: str
    status_detail: str
    chunks_count: int
    char_length: int


class DocumentAnalysisResponse(BaseResponse):
    char_length: int
    word_count: int
    legal_feature_scores: Dict[str, float]
    extracted_acts: List[str]
    extracted_sections: List[str]


class ContractRiskResponse(BaseResponse):
    risk_score: float
    risk_level: str
    engineered_features: Dict[str, float]


class RecommendationResponse(BaseResponse):
    user_id: str
    recommendations: List[str]


class SummaryResponse(BaseResponse):
    summary: str


class HistoryResponse(BaseResponse):
    session_id: str
    history: List[ChatMessage]


class HistoryClearResponse(BaseResponse):
    session_id: str
    cleared: bool


# Telemetry Diagnostic responses

class HealthResponse(BaseModel):
    status: str = Field("healthy")


class MetricsResponse(BaseModel):
    total_requests: int
    error_requests: int
    average_latency_ms: float
    route_requests: Dict[str, int]


class StatusResponse(BaseModel):
    status: str
    database: DatabaseStats
    models: ModelCacheStats
    resources: ResourceStats
