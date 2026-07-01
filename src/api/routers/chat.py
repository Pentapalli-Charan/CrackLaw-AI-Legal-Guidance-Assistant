import time
import json
from fastapi import APIRouter, Depends, Query, Security
from fastapi.responses import StreamingResponse
from src.api.dependencies import get_chat_service, get_history_service, verify_api_key
from src.services.chat_service import ChatService
from src.services.history_service import HistoryService
from src.api.models.request_models import ChatRequest
from src.api.models.response_models import ChatResponse, HistoryResponse, HistoryClearResponse

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat_message(
    body: ChatRequest,
    chat_svc: ChatService = Depends(get_chat_service),
    api_key: str = Security(verify_api_key)
):
    """Submits query to the reasoning pipeline, returning citations and confidence diagnostics."""
    t0 = time.time()
    result = chat_svc.send_message(body.session_id, body.query, body.options)
    latency = (time.time() - t0) * 1000
    
    # Pack latency
    result["latency_ms"] = latency
    result["status"] = "success"
    return result


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    chat_svc: ChatService = Depends(get_chat_service),
    api_key: str = Security(verify_api_key)
):
    """Streams token events in Server-Sent Events (SSE) format, ending with the full metadata block."""
    def event_generator():
        try:
            generator = chat_svc.stream_message(body.session_id, body.query, body.options)
            for chunk in generator:
                # Format chunk strictly according to SSE specification
                event_name = chunk.get("event", "message")
                data_str = json.dumps(chunk.get("data", {}))
                yield f"event: {event_name}\ndata: {data_str}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history", response_model=HistoryResponse)
async def chat_history(
    session_id: str = Query(..., min_length=1, description="Chat session thread ID to load."),
    history_svc: HistoryService = Depends(get_history_service),
    api_key: str = Security(verify_api_key)
):
    """Retrieves conversation memory logs for the requested session thread."""
    t0 = time.time()
    history = history_svc.get_chat_history(session_id)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "session_id": session_id,
        "history": history
    }


@router.delete("/history", response_model=HistoryClearResponse)
async def clear_history(
    session_id: str = Query(..., min_length=1, description="Chat session thread ID to delete."),
    history_svc: HistoryService = Depends(get_history_service),
    api_key: str = Security(verify_api_key)
):
    """Purges the cached session memory registers."""
    t0 = time.time()
    cleared = history_svc.clear_chat_history(session_id)
    latency = (time.time() - t0) * 1000
    
    return {
        "status": "success",
        "latency_ms": latency,
        "session_id": session_id,
        "cleared": cleared
    }
