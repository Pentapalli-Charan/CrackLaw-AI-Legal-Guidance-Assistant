import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("CrackLaw.AI.ResponseFormatter")

class ResponseFormatter:
    """Parses raw text outputs from the LLM and reformats them into structured legal responses."""

    def __init__(self):
        pass

    def _listify_content(self, content: str) -> List[str]:
        """Converts bulleted/numbered markdown lists into lists of strings."""
        if not content:
            return []
        lines = content.split("\n")
        items = []
        for line in lines:
            # Strip list symbols, bullet marks, leading numbers, and spaces
            cleaned = re.sub(r"^[#\s\-*\d+\.\)]*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Splits raw text response by headings using a robust regex search."""
        sections = {}
        headers_map = {
            "SUMMARY": "summary",
            "RELEVANT ACTS": "relevant_acts",
            "RELEVANT SECTIONS": "relevant_sections",
            "SUPPORTING CITATIONS": "supporting_citations",
            "KEY POINTS": "key_points",
            "SUGGESTED NEXT STEPS": "suggested_next_steps"
        }

        lines = text.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            # Strip leading markdown symbols for classification
            stripped_line = re.sub(r"^[#\s\-*]*", "", line).strip().upper()
            
            matched_section = None
            matched_header_key = None
            for h_key in headers_map:
                if stripped_line == h_key:
                    matched_section = headers_map[h_key]
                    matched_header_key = h_key
                    break
                elif stripped_line.startswith(h_key + ":"):
                    matched_section = headers_map[h_key]
                    matched_header_key = h_key
                    break
                elif stripped_line.startswith(h_key + " "):
                    matched_section = headers_map[h_key]
                    matched_header_key = h_key
                    break

            if matched_section:
                # Save previous section content
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = matched_section
                current_content = []
                
                # Check for inline text on the same line after the header name (e.g., "SUMMARY: This is text")
                header_regex = re.compile(rf"^\s*[#\-\s\*]*{re.escape(matched_header_key)}\s*:?\s*", re.IGNORECASE)
                match = header_regex.match(line)
                if match:
                    inline_text = line[match.end():].strip()
                    if inline_text:
                        current_content.append(inline_text)
            else:
                if current_section:
                    current_content.append(line)
                else:
                    # Content before any header defaults to Summary
                    if line.strip():
                        current_section = "summary"
                        current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def format_response(
        self,
        raw_response: str,
        citations: List[Dict[str, Any]],
        confidence_score: float,
        disclaimer: str
    ) -> Dict[str, Any]:
        """Assembles the raw response text, citations, and metrics into a standard schema dictionary."""
        
        parsed_sections = self._parse_sections(raw_response)

        # Build list of citations formatted as strings
        citation_strings = [
            c.get("formatted_citation") or f"{c.get('act_name')} Section {c.get('section')}"
            for c in citations
        ]

        # Extract values with fallbacks
        summary = parsed_sections.get("summary", raw_response).strip()
        relevant_acts = self._listify_content(parsed_sections.get("relevant_acts", ""))
        relevant_sections = self._listify_content(parsed_sections.get("relevant_sections", ""))
        key_points = self._listify_content(parsed_sections.get("key_points", ""))
        suggested_next_steps = self._listify_content(parsed_sections.get("suggested_next_steps", ""))
        
        # Merge supporting citations parsed from LLM text with generated structured citations list
        supporting_citations = self._listify_content(parsed_sections.get("supporting_citations", ""))
        for c_str in citation_strings:
            if c_str and c_str not in supporting_citations:
                supporting_citations.append(c_str)

        # Fallback list generation based on structured citations list if sections are missing
        if not relevant_acts and citations:
            relevant_acts = list({c["act_name"] for c in citations if c.get("act_name")})
        if not relevant_sections and citations:
            relevant_sections = list({f"Section {c['section']}" for c in citations if c.get("section")})

        structured_data = {
            "summary": summary,
            "relevant_acts": relevant_acts,
            "relevant_sections": relevant_sections,
            "supporting_citations": supporting_citations,
            "key_points": key_points,
            "suggested_next_steps": suggested_next_steps,
            "disclaimer": disclaimer,
            "confidence_score": confidence_score
        }

        logger.debug("Formatted response structured keys: %s", list(structured_data.keys()))
        return structured_data
