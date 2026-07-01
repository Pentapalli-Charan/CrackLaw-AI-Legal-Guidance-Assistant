import time
from fastapi import APIRouter, Depends, Security
from src.api.dependencies import get_contract_service, verify_api_key
from src.services.contract_service import ContractService
from src.api.models.request_models import ContractRiskRequest
from src.api.models.response_models import ContractRiskResponse

router = APIRouter(prefix="/api/v1/contracts", tags=["Contracts"])

@router.post("/analyze", response_model=ContractRiskResponse)
async def analyze_contract(
    body: ContractRiskRequest,
    contract_svc: ContractService = Depends(get_contract_service),
    api_key: str = Security(verify_api_key)
):
    """Calculates risk levels and displays engineered model indices for contract texts."""
    t0 = time.time()
    res = contract_svc.analyze_contract(body.text)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "risk_score": res.get("risk_score", 0.0),
        "risk_level": res.get("risk_level", "LOW"),
        "engineered_features": res.get("features", {})
    }
