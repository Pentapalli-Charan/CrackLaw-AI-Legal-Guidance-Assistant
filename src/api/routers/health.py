from fastapi import APIRouter, Depends
from src.api.dependencies import get_health_service
from src.services.health_service import HealthService
from src.api.models.response_models import HealthResponse, MetricsResponse, StatusResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def check_health(health_svc: HealthService = Depends(get_health_service)):
    """Simple ping-pong health endpoint verifying app availability."""
    return health_svc.get_health()


@router.get("/metrics", response_model=MetricsResponse)
async def check_metrics(health_svc: HealthService = Depends(get_health_service)):
    """Returns rolling requests, error counts, and average latency telemetry."""
    return health_svc.get_metrics()


@router.get("/status", response_model=StatusResponse)
async def check_status(health_svc: HealthService = Depends(get_health_service)):
    """Diagnoses memory caches, DB indices length, and server hardware utilization."""
    return health_svc.get_status()
