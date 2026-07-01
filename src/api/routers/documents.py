import time
from fastapi import APIRouter, Depends, UploadFile, File, Security
from src.api.dependencies import get_document_service, get_summary_service, verify_api_key
from src.services.document_service import DocumentService
from src.services.summary_service import SummaryService
from src.api.models.request_models import DocumentAnalysisRequest, SummaryRequest
from src.api.models.response_models import DocumentUploadResponse, DocumentAnalysisResponse, SummaryResponse

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_svc: DocumentService = Depends(get_document_service),
    api_key: str = Security(verify_api_key)
):
    """Parses, chunks, embeds, and indexes an uploaded PDF, DOCX, TXT, or Image document."""
    t0 = time.time()
    content = await file.read()
    
    # Run upload, chunking, and database indexing pipeline
    res = doc_svc.upload_document(file.filename, content)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "document_id": res["document_id"],
        "filename": res["filename"],
        "status_detail": res["status"],
        "chunks_count": res["chunks_count"],
        "char_length": res["char_length"]
    }


@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    body: DocumentAnalysisRequest,
    doc_svc: DocumentService = Depends(get_document_service),
    api_key: str = Security(verify_api_key)
):
    """Extracts structural parts, acts, and keyword metrics from raw document text."""
    t0 = time.time()
    res = doc_svc.analyze_document(body.text)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "char_length": res["char_length"],
        "word_count": res["word_count"],
        "legal_feature_scores": res["legal_feature_scores"],
        "extracted_acts": res["extracted_acts"],
        "extracted_sections": res["extracted_sections"]
    }


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_document_contents(
    body: SummaryRequest,
    summary_svc: SummaryService = Depends(get_summary_service),
    api_key: str = Security(verify_api_key)
):
    """Produces a concise summary of long document texts using the LLM gateway."""
    t0 = time.time()
    summary = summary_svc.summarize_text(body.text, body.max_length)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "summary": summary
    }
