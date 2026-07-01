import re
import logging
from typing import List, Dict, Any
from src.retrieval.search_result import SearchResultItem

logger = logging.getLogger("CrackLaw.AI.CitationGenerator")

class CitationGenerator:
    """Extracts, verifies, and formats supporting legal citations from retrieved metadata and response texts."""

    def __init__(self):
        # Match brackets like [1], [Document 1], [Doc 1], [doc1]
        self.ref_patterns = [
            re.compile(r"\[Document\s+(\d+)\]", re.IGNORECASE),
            re.compile(r"\[Doc\s+(\d+)\]", re.IGNORECASE),
            re.compile(r"\[(\d+)\]")
        ]

    def _extract_referenced_indices(self, text: str) -> List[int]:
        """Finds bracketed document indices in the response text."""
        indices = set()
        for pattern in self.ref_patterns:
            for match in pattern.finditer(text):
                try:
                    idx = int(match.group(1))
                    indices.add(idx)
                except ValueError:
                    continue
        return sorted(list(indices))

    def generate_citations(self, results: List[SearchResultItem], response_text: str) -> List[Dict[str, Any]]:
        """Scans response text for citation indicators and aligns them with retrieved chunk metadata."""
        if not results:
            return []

        citations = []
        referenced_indices = self._extract_referenced_indices(response_text)
        
        for idx, item in enumerate(results, start=1):
            meta = item.metadata or {}
            
            # Resolve citation fields
            act = meta.get("act") or getattr(item, "act", "") or ""
            chapter = meta.get("chapter") or getattr(item, "chapter", "") or ""
            section = meta.get("section") or getattr(item, "section", "") or ""
            subsection = meta.get("subsection") or getattr(item, "subsection", "") or ""
            judgment_name = meta.get("judgment") or meta.get("judgment_name") or getattr(item, "judgment", "") or ""
            source = meta.get("source") or getattr(item, "source", "") or item.document_id
            citation_str = item.citation or f"{act}, Section {section}" if act and section else ""

            # Check if this document index was explicitly referenced
            is_directly_cited = (idx in referenced_indices)

            # Rule-based fallback: check if response text mentions the Act name, section number, or judgment name
            is_lexically_cited = False
            if not is_directly_cited:
                if act and len(act) > 3 and act.lower() in response_text.lower():
                    is_lexically_cited = True
                if section and section.lower() in response_text.lower():
                    # Avoid false positives like matching "1" inside "10"
                    sec_pattern = re.compile(rf"\b{re.escape(section.lower())}\b")
                    if sec_pattern.search(response_text.lower()):
                        is_lexically_cited = True
                if judgment_name and len(judgment_name) > 3 and judgment_name.lower() in response_text.lower():
                    is_lexically_cited = True

            # If it is referenced directly or via keywords, compile structured citation
            if is_directly_cited or is_lexically_cited:
                citation_record = {
                    "citation_id": f"cit_{item.chunk_id}",
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "act_name": act,
                    "chapter": chapter,
                    "section": section,
                    "subsection": subsection,
                    "judgment_name": judgment_name,
                    "document_source": source,
                    "formatted_citation": citation_str,
                    "reference_type": "explicit" if is_directly_cited else "implicit"
                }
                citations.append(citation_record)

        # If no citations were detected but we had chunks, include the top chunk as general support
        if not citations and results:
            item = results[0]
            meta = item.metadata or {}
            act = meta.get("act") or "N/A"
            section = meta.get("section") or "N/A"
            citations.append({
                "citation_id": f"cit_{item.chunk_id}",
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "act_name": act,
                "chapter": meta.get("chapter") or "N/A",
                "section": section,
                "subsection": meta.get("subsection") or "N/A",
                "judgment_name": meta.get("judgment") or meta.get("judgment_name") or "N/A",
                "document_source": meta.get("source") or item.document_id,
                "formatted_citation": item.citation or f"{act}, Section {section}",
                "reference_type": "contextual"
            })

        logger.info("Generated %d citations for response.", len(citations))
        return citations
