import time
from fastapi import APIRouter, Depends, Security
from src.api.dependencies import get_summary_service, verify_api_key
from src.services.summary_service import SummaryService
from src.api.models.request_models import SummaryRequest
from src.api.models.response_models import SummaryResponse

router = APIRouter(prefix="/api/v1/summary", tags=["Summary"])

@router.post("", response_model=SummaryResponse)
async def summarize_text(
    body: SummaryRequest,
    summary_svc: SummaryService = Depends(get_summary_service),
    api_key: str = Security(verify_api_key)
):
    """Generates a concise legal summary for text bodies."""
    t0 = time.time()
    summary = summary_svc.summarize_text(body.text, body.max_length)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "summary": summary
    }
