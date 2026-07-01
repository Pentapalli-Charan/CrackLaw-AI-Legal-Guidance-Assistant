import os
import json
import logging
import threading
from typing import Dict, Any, Optional
from src.ai.conversation_memory import ConversationMemory
from src.ai.token_manager import TokenManager
from src.ai.ai_models import Message

logger = logging.getLogger("CrackLaw.AI.SessionManager")

class SessionManager:
    """Thread-safe orchestrator managing chat session objects and memory lifecycles."""

    def __init__(self, token_manager: Optional[TokenManager] = None, max_history_tokens: int = 4096):
        self.token_manager = token_manager or TokenManager()
        self.max_history_tokens = max_history_tokens
        # Mapping: { session_id -> { "memory": ConversationMemory, "metadata": Dict } }
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieves or creates a session context container (memory and metadata) thread-safely."""
        with self.lock:
            if session_id not in self.sessions:
                logger.info("Initializing new chat session: %s", session_id)
                self.sessions[session_id] = {
                    "memory": ConversationMemory(
                        token_manager=self.token_manager,
                        max_history_tokens=self.max_history_tokens
                    ),
                    "metadata": {}
                }
            return self.sessions[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Removes a session context from the active pool."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info("Evicted session from active pool: %s", session_id)
                return True
            return False

    def clear_all(self) -> None:
        """Evicts all active sessions."""
        with self.lock:
            self.sessions.clear()
            logger.info("All session memories cleared.")

    def export_sessions_to_json(self, filepath: str) -> None:
        """Serializes current session pool details to a JSON file for persistence."""
        with self.lock:
            exported_data = {}
            for sid, container in self.sessions.items():
                messages = container["memory"].get_messages()
                exported_data[sid] = {
                    "history": [msg.to_dict() for msg in messages],
                    "metadata": container["metadata"]
                }
            
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(exported_data, f, ensure_ascii=False, indent=2)
                logger.info("Exported %d sessions to file: %s", len(exported_data), filepath)
            except Exception as e:
                logger.error("Failed to export sessions: %s", str(e))

    def import_sessions_from_json(self, filepath: str) -> None:
        """Restores session registers from a persisted JSON file."""
        if not os.path.exists(filepath):
            logger.warning("Session backup file not found for restore: %s", filepath)
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                imported_data = json.load(f)
            
            with self.lock:
                for sid, data in imported_data.items():
                    memory = ConversationMemory(
                        token_manager=self.token_manager,
                        max_history_tokens=self.max_history_tokens
                    )
                    # Load message models
                    for msg_dict in data.get("history", []):
                        memory.messages.append(Message.from_dict(msg_dict))
                    
                    self.sessions[sid] = {
                        "memory": memory,
                        "metadata": data.get("metadata", {})
                    }
                logger.info("Imported %d sessions from file: %s", len(imported_data), filepath)
        except Exception as e:
            logger.error("Failed to import sessions: %s", str(e))
