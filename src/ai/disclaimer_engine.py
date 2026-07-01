import logging
from typing import Dict, Optional

logger = logging.getLogger("CrackLaw.AI.DisclaimerEngine")

class DisclaimerEngine:
    """Manages and appends appropriate legal disclaimers to AI-generated responses."""

    def __init__(self, templates: Optional[Dict[str, str]] = None):
        # Default disclaimers for different intents
        self.templates = templates or {
            "default": (
                "Disclaimer: CrackLaw is an AI-powered legal guidance assistant. "
                "The information provided is for educational and informational purposes only "
                "and does not constitute formal legal advice. No attorney-client relationship "
                "is created. Consult a qualified legal advocate for official representation."
            ),
            "Contract Review": (
                "Disclaimer: This automated contract analysis identifies key terms and potential risks "
                "based on the text. It does not constitute a formal legal audit or opinion. "
                "Always review clauses with a qualified contract attorney before execution."
            ),
            "Judgment Summary": (
                "Disclaimer: This judgment summary is auto-generated using AI semantic parsing. "
                "While we strive for accuracy, case summaries can omit critical contextual elements. "
                "Verify facts directly against official court transcripts before citing in court pleadings."
            ),
            "Case Search": (
                "Disclaimer: This judgment summary is auto-generated using AI semantic parsing. "
                "While we strive for accuracy, case summaries can omit critical contextual elements. "
                "Verify facts directly against official court transcripts before citing in court pleadings."
            ),
            "Legal Risk Analysis": (
                "Disclaimer: This risk assessment is automated and focuses on textual compliance flags. "
                "It does not cover all regulatory, operational, or jurisdictional liabilities. "
                "Seek professional counsel to address specific compliance exposure."
            )
        }

    def get_disclaimer(self, intent: str) -> str:
        """Retrieves the disclaimer matching the intent, falling back to the default disclaimer."""
        disclaimer = self.templates.get(intent)
        if not disclaimer:
            disclaimer = self.templates.get("default")
        
        logger.debug("Disclaimer resolved for intent '%s'", intent)
        return disclaimer
