import os
from typing import Optional
from fastapi import Header, Query, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader, APIKeyQuery
from src.config import Config
from src.services.service_factory import ServiceFactory

# Services
from src.services.chat_service import ChatService
from src.services.retrieval_service import RetrievalServiceWrapper
from src.services.document_service import DocumentService
from src.services.contract_service import ContractService
from src.services.summary_service import SummaryService
from src.services.recommendation_service import RecommendationService
from src.services.history_service import HistoryService
from src.services.health_service import HealthService

# Get Shared ServiceFactory instance
_factory = ServiceFactory()

def get_service_factory() -> ServiceFactory:
    """Returns the singleton ServiceFactory instance."""
    return _factory

def get_chat_service(factory: ServiceFactory = Depends(get_service_factory)) -> ChatService:
    return factory.get_chat_service()

def get_retrieval_service(factory: ServiceFactory = Depends(get_service_factory)) -> RetrievalServiceWrapper:
    return factory.get_retrieval_service()

def get_document_service(factory: ServiceFactory = Depends(get_service_factory)) -> DocumentService:
    return factory.get_document_service()

def get_contract_service(factory: ServiceFactory = Depends(get_service_factory)) -> ContractService:
    return factory.get_contract_service()

def get_summary_service(factory: ServiceFactory = Depends(get_service_factory)) -> SummaryService:
    return factory.get_summary_service()

def get_recommendation_service(factory: ServiceFactory = Depends(get_service_factory)) -> RecommendationService:
    return factory.get_recommendation_service()

def get_history_service(factory: ServiceFactory = Depends(get_service_factory)) -> HistoryService:
    return factory.get_history_service()

def get_health_service(factory: ServiceFactory = Depends(get_service_factory)) -> HealthService:
    return factory.get_health_service()


# --- API Key Authentication Gateway ---

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

def verify_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    api_key_query: Optional[str] = Security(API_KEY_QUERY)
) -> str:
    """Validates API Key using environment or config. Passes gracefully if authentication is disabled."""
    # Fetch expected API key from environment variables (fallback: empty = authentication disabled)
    expected_key = os.environ.get("CRACKLAW_API_KEY")
    if not expected_key:
        return "auth_disabled"

    token = api_key_header or api_key_query
    if not token or token != expected_key:
        from src.services.exceptions import SecurityError as ServiceSecurityError
        raise ServiceSecurityError(
            "Unauthorized: Access requires a valid 'X-API-Key' header or 'api_key' query parameter."
        )
    return token
