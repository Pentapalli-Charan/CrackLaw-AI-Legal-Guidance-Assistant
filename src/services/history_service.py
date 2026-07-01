import logging
from typing import List, Dict, Any, Optional
from src.ai.session_manager import SessionManager
from src.services.exceptions import ServiceError, NotFoundError

logger = logging.getLogger("CrackLaw.Services.HistoryService")

class HistoryService:
    """Manages conversational session logs, serialization, and clearance."""

    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or SessionManager()

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves list of chat messages for a specific session."""
        try:
            logger.info("Fetching history for session %s", session_id)
            # Fetch session container (creates one if not exists)
            container = self.session_manager.get_session(session_id)
            memory = container["memory"]
            messages = memory.get_messages()
            
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": getattr(msg, "timestamp", None)
                }
                for msg in messages
            ]
        except Exception as e:
            logger.error("Failed to retrieve chat history for session %s: %s", session_id, str(e))
            raise ServiceError(f"Failed to fetch chat history: {e}") from e

    def clear_chat_history(self, session_id: str) -> bool:
        """Purges conversational history for a specific session."""
        try:
            logger.info("Clearing chat history for session %s", session_id)
            evicted = self.session_manager.delete_session(session_id)
            if not evicted:
                logger.warning("Attempted to delete non-existent session: %s", session_id)
            return True
        except Exception as e:
            logger.error("Failed to clear chat history for session %s: %s", session_id, str(e))
            raise ServiceError(f"Failed to clear chat history: {e}") from e
