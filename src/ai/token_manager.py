import logging
from typing import List, Optional
from src.ai.ai_models import Message

logger = logging.getLogger("CrackLaw.AI.TokenManager")

class TokenManager:
    """Estimates and manages token budgeting for messages, prompts, and context blocks."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self._initialized = False

        # Attempt to load a tokenizer from Hugging Face transformers
        try:
            from transformers import AutoTokenizer
            # Suppress excessive logging during download attempt
            logging.getLogger("transformers").setLevel(logging.WARNING)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self._initialized = True
            logger.info("TokenManager initialized with local tokenizer: %s", model_name)
        except Exception:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._initialized = True
                logger.info("TokenManager initialized with Hugging Face online tokenizer: %s", model_name)
            except Exception as e:
                logger.warning(
                    "Could not load Hugging Face tokenizer '%s' (%s). Fallback token estimator will be used.",
                    model_name,
                    str(e)
                )

    def count_tokens(self, text: str) -> int:
        """Estimates or counts the number of tokens in a string."""
        if not text:
            return 0

        if self._initialized and self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.debug("Tokenizer encoding failed, using fallback: %s", str(e))

        # Fallback estimation: average of word-based (~1.3 tokens per word) and character-based (~4 chars per token)
        words = text.split()
        word_estimate = int(len(words) * 1.3)
        char_estimate = int(len(text) / 4)
        return max(1, (word_estimate + char_estimate) // 2)

    def count_message_tokens(self, messages: List[Message]) -> int:
        """Calculates tokens used by a list of messages, accounting for message structural overhead."""
        num_tokens = 0
        for msg in messages:
            # Overheads for chat formats (approx 4 tokens per message block)
            num_tokens += 4
            num_tokens += self.count_tokens(msg.content)
            num_tokens += self.count_tokens(msg.role)
        num_tokens += 2  # priming tokens for the assistant response
        return num_tokens

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within a given token budget."""
        if not text or max_tokens <= 0:
            return ""

        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # If tokenizer is available, perform exact token-level truncation
        if self._initialized and self.tokenizer is not None:
            try:
                encoded = self.tokenizer.encode(text)
                truncated_encoded = encoded[:max_tokens]
                return self.tokenizer.decode(truncated_encoded)
            except Exception as e:
                logger.debug("Tokenizer truncation failed, using fallback: %s", str(e))

        # Fallback character-based truncation
        # Start with a rough proportional slice of character length, then adjust
        char_ratio = len(text) / max(1, current_tokens)
        target_char_len = int(max_tokens * char_ratio)
        
        truncated = text[:target_char_len]
        # Fine-tune truncation length to ensure safety
        while self.count_tokens(truncated) > max_tokens and len(truncated) > 0:
            # Back off by words
            words = truncated.split()
            if not words:
                break
            truncated = " ".join(words[:-1])

        return truncated
