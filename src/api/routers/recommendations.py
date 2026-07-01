import time
from fastapi import APIRouter, Depends, Security
from src.api.dependencies import get_recommendation_service, verify_api_key
from src.services.recommendation_service import RecommendationService
from src.api.models.request_models import RecommendationRequest
from src.api.models.response_models import RecommendationResponse

router = APIRouter(prefix="/api/v1/recommendations", tags=["Recommendations"])

@router.post("", response_model=RecommendationResponse)
async def get_recommendations(
    body: RecommendationRequest,
    rec_svc: RecommendationService = Depends(get_recommendation_service),
    api_key: str = Security(verify_api_key)
):
    """Retrieves case/act recommendations for the requested user profile."""
    t0 = time.time()
    recs = rec_svc.get_recommendations(body.user_id, body.user_history)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "user_id": body.user_id,
        "recommendations": recs
    }
