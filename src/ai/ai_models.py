import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Message:
    """Represents a single message in the conversation history."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time())
        )


@dataclass
class Session:
    """Represents a conversational session with message history and metadata."""
    session_id: str
    history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "history": [msg.to_dict() for msg in self.history],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        history = [Message.from_dict(m) for m in data.get("history", [])]
        return cls(
            session_id=data["session_id"],
            history=history,
            metadata=data.get("metadata", {})
        )


@dataclass
class AIRequest:
    """Incoming request to the AI Engine."""
    query: str
    session_id: str
    options: Dict[str, Any] = field(default_factory=dict)  # e.g., provider overrides, temperature


@dataclass
class AIResponse:
    """Final response returned by the AI Engine containing the structured answer, citations, and metrics."""
    response_text: str
    intent: str
    rewritten_query: str
    retrieved_context: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    validation_result: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_text": self.response_text,
            "intent": self.intent,
            "rewritten_query": self.rewritten_query,
            "retrieved_context": self.retrieved_context,
            "structured_data": self.structured_data,
            "citations": self.citations,
            "confidence_score": self.confidence_score,
            "validation_result": self.validation_result,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "model": self.model
        }
