import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("CrackLaw.Retrieval.MetadataFilter")

class MetadataFilter:
    """Validates, sanitizes, and normalizes filter requests passed to retrieval search engines."""

    # Set of supported database index and JSONB sub-keys
    ALLOWED_KEYS = {
        "document_id",
        "act",
        "chapter",
        "section",
        "subsection",
        "language",
        "source",
        "version",
        "doc_type",
        "document_type",
        "jurisdiction",
        "date"
    }

    @staticmethod
    def clean_filters(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validates filter dict, stripping unmapped keys or values with empty properties."""
        if not filters:
            return {}

        cleaned = {}
        for k, v in filters.items():
            if v is None or v == "":
                continue

            k_clean = k.strip().lower()
            if k_clean in MetadataFilter.ALLOWED_KEYS:
                # Map alternate keys to schema equivalents
                if k_clean == "document_type":
                    cleaned["doc_type"] = v
                else:
                    cleaned[k_clean] = v
            else:
                logger.warning("Stripped unsupported filter key request: '%s'", k)

        return cleaned
