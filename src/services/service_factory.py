import logging
from typing import Optional

from src.config import Config
from src.retrieval.retrieval_service import RetrievalService
from src.ai.ai_service import LegalAIService
from src.models.model_hub import ModelHub

# Import Wrapper/Service classes
from src.services.chat_service import ChatService
from src.services.retrieval_service import RetrievalServiceWrapper
from src.services.document_service import DocumentService
from src.services.contract_service import ContractService
from src.services.summary_service import SummaryService
from src.services.recommendation_service import RecommendationService
from src.services.history_service import HistoryService
from src.services.health_service import HealthService

logger = logging.getLogger("CrackLaw.Services.ServiceFactory")

class ServiceFactory:
    """Central factory for initializing and injecting dependencies into service classes."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ServiceFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Config] = None):
        if self._initialized:
            return
            
        logger.info("Initializing ServiceFactory singleton...")
        self.config = config or Config()
        
        # Instantiate core shared singletons
        self.retrieval_service = RetrievalService(self.config)
        self.model_hub = ModelHub()
        
        # Setup Legal AI Engine facade
        self.ai_service = LegalAIService(
            config=self.config,
            retrieval_service=self.retrieval_service
        )
        
        # Instantiate wrapper services
        self.chat_svc = ChatService(self.ai_service)
        self.retrieval_svc = RetrievalServiceWrapper(self.retrieval_service)
        self.doc_svc = DocumentService(self.config)
        self.contract_svc = ContractService(self.model_hub)
        self.summary_svc = SummaryService(self.ai_service)
        self.rec_svc = RecommendationService(self.model_hub)
        self.history_svc = HistoryService(self.ai_service.session_manager)
        self.health_svc = HealthService(self.config, self.model_hub)

        self._initialized = True
        logger.info("ServiceFactory singleton initialized successfully.")

    def get_chat_service(self) -> ChatService:
        return self.chat_svc

    def get_retrieval_service(self) -> RetrievalServiceWrapper:
        return self.retrieval_svc

    def get_document_service(self) -> DocumentService:
        return self.doc_svc

    def get_contract_service(self) -> ContractService:
        return self.contract_svc

    def get_summary_service(self) -> SummaryService:
        return self.summary_svc

    def get_recommendation_service(self) -> RecommendationService:
        return self.rec_svc

    def get_history_service(self) -> HistoryService:
        return self.history_svc

    def get_health_service(self) -> HealthService:
        return self.health_svc
