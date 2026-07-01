import logging
from typing import Dict, Any, List, Optional
from src.ai.prompt_templates import (
    BASE_SYSTEM_PROMPT,
    INTENT_DETECTION_TEMPLATE,
    QUERY_REWRITING_TEMPLATE,
    TEMPLATE_MAP
)

logger = logging.getLogger("CrackLaw.AI.PromptEngine")

class PromptEngine:
    """Assembles prompt payloads dynamically by inserting context, history, and queries into templates."""

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = templates or TEMPLATE_MAP

    def build_system_prompt(self, context: str) -> str:
        """Assembles the core system instructions with the retrieved context injected."""
        return BASE_SYSTEM_PROMPT.format(context=context or "No retrieved legal context available.")

    def build_user_prompt(self, query: str, intent: str) -> str:
        """Finds the prompt template corresponding to the classified intent and formats it with the query."""
        template = self.templates.get(intent)
        if not template:
            # Default fallback to Legal Information guidance
            template = self.templates.get("Legal Information")
            
        try:
            return template.format(query=query)
        except Exception as e:
            logger.error("Failed to format user prompt template for intent '%s': %s", intent, str(e))
            return f"User Query: {query}"

    def build_intent_detection_prompt(self, query: str) -> str:
        """Creates the prompt text for classifying user query intent."""
        return INTENT_DETECTION_TEMPLATE.format(query=query)

    def build_query_rewriter_prompt(self, query: str, history_str: str) -> str:
        """Creates the prompt text to rewrite and resolve co-references in query."""
        return QUERY_REWRITING_TEMPLATE.format(query=query, history=history_str or "No history.")
