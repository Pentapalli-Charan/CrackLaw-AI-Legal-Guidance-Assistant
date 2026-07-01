import time
from fastapi import APIRouter, Depends, Security
from src.api.dependencies import get_retrieval_service, verify_api_key
from src.services.retrieval_service import RetrievalServiceWrapper
from src.api.models.request_models import SearchQueryRequest
from src.api.models.response_models import SearchResponse

router = APIRouter(prefix="/api/v1/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
async def search(
    body: SearchQueryRequest,
    search_svc: RetrievalServiceWrapper = Depends(get_retrieval_service),
    api_key: str = Security(verify_api_key)
):
    """Executes a hybrid search query with automated cache matching."""
    t0 = time.time()
    res = search_svc.search(body.query, body.filters, "hybrid", body.top_k, body.min_similarity)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "query": res["query"],
        "results": res["results"],
        "context": res["context"]
    }


@router.post("/semantic", response_model=SearchResponse)
async def search_semantic(
    body: SearchQueryRequest,
    search_svc: RetrievalServiceWrapper = Depends(get_retrieval_service),
    api_key: str = Security(verify_api_key)
):
    """Retrieves context snippets using dense vector similarity."""
    t0 = time.time()
    res = search_svc.search_semantic(body.query, body.filters, body.top_k, body.min_similarity)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "query": res["query"],
        "results": res["results"],
        "context": res["context"]
    }


@router.post("/hybrid", response_model=SearchResponse)
async def search_hybrid(
    body: SearchQueryRequest,
    search_svc: RetrievalServiceWrapper = Depends(get_retrieval_service),
    api_key: str = Security(verify_api_key)
):
    """Retrieves content combining keyword counts and embedding similarity indices."""
    t0 = time.time()
    res = search_svc.search_hybrid(body.query, body.filters, body.top_k, body.min_similarity)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "query": res["query"],
        "results": res["results"],
        "context": res["context"]
    }
