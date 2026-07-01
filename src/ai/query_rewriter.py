import logging
from typing import List, Optional, Any
from src.ai.ai_models import Message

logger = logging.getLogger("CrackLaw.AI.QueryRewriter")

class QueryRewriter:
    """Uses LLM reasoning to rewrite vague or context-dependent queries into standalone retrieval-friendly terms."""

    def __init__(self, llm_gateway: Any, prompt_engine: Any):
        self.llm_gateway = llm_gateway
        self.prompt_engine = prompt_engine

    def _format_history(self, history: List[Message], max_turns: int = 5) -> str:
        """Formats the last N turns of message history into a readable transcript for prompt rewriting."""
        if not history:
            return "No prior conversation history."
            
        recent_history = history[-max_turns * 2:]  # Last N turns (user + assistant)
        transcript = []
        for msg in recent_history:
            role_label = "User" if msg.role == "user" else "Assistant"
            transcript.append(f"{role_label}: {msg.content}")
            
        return "\n".join(transcript)

    def rewrite_query(self, query: str, history: List[Message]) -> str:
        """Translates the query to make it context-complete for vector searches."""
        if not history:
            logger.debug("No history found; skipping query rewriting.")
            return query

        history_str = self._format_history(history)
        prompt = self.prompt_engine.build_query_rewriter_prompt(query, history_str)

        try:
            rewritten = self.llm_gateway.generate(
                prompt=prompt,
                system_prompt="You are a query rewriting model. Output ONLY the standalone search query.",
                temperature=0.1,
                max_tokens=64
            )
            rewritten_query = rewritten.strip().replace('"', '').replace("'", "")
            
            if not rewritten_query or len(rewritten_query) < 2:
                logger.warning("Query rewriter returned empty or invalid rewrite. Using original.")
                return query
                
            logger.info("Query rewritten: '%s' -> '%s'", query, rewritten_query)
            return rewritten_query
        except Exception as e:
            logger.error("Query rewriting failed (falling back to original query): %s", str(e))
            return query
