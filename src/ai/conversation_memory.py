import logging
from typing import List, Optional
from src.ai.ai_models import Message
from src.ai.token_manager import TokenManager

logger = logging.getLogger("CrackLaw.AI.ConversationMemory")

class ConversationMemory:
    """Manages an in-memory sliding conversation window, ensuring it respects model token budgets."""

    def __init__(self, token_manager: Optional[TokenManager] = None, max_history_tokens: int = 4096):
        self.token_manager = token_manager or TokenManager()
        self.max_history_tokens = max_history_tokens
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the active history and triggers the sliding truncation check."""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.trim_history()

    def get_messages(self) -> List[Message]:
        """Returns the list of active messages in the conversation session."""
        return self.messages

    def get_history_as_string(self) -> str:
        """Serializes current dialog history into a single text transcript."""
        transcript = []
        for msg in self.messages:
            if msg.role == "system":
                continue
            role_label = "User" if msg.role == "user" else "Assistant"
            transcript.append(f"{role_label}: {msg.content}")
        return "\n".join(transcript)

    def trim_history(self) -> None:
        """Trims message history from the oldest entries (excluding system prompts) if token budget is exceeded."""
        current_tokens = self.token_manager.count_message_tokens(self.messages)
        if current_tokens <= self.max_history_tokens:
            return

        logger.info(
            "Conversation memory size (%d tokens) exceeds threshold (%d). Trimming...",
            current_tokens, self.max_history_tokens
        )

        # Separate system messages and chat messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        chat_msgs = [m for m in self.messages if m.role != "system"]

        # Keep removing from the oldest chat messages until budget is satisfied
        while chat_msgs and self.token_manager.count_message_tokens(system_msgs + chat_msgs) > self.max_history_tokens:
            removed = chat_msgs.pop(0)
            logger.debug("Evicted message from memory: '%s: %s...'", removed.role, removed.content[:20])

        self.messages = system_msgs + chat_msgs
        logger.debug(
            "Trimmed conversation memory successfully. New token count: %d",
            self.token_manager.count_message_tokens(self.messages)
        )

    def clear(self) -> None:
        """Resets the message stack."""
        self.messages.clear()
        logger.debug("Conversation memory cleared.")
