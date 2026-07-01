import os
import re
import logging
from typing import Optional, Dict, List, Any
from src.config import Config
from src.metadata import MetadataManager

logger = logging.getLogger("CrackLaw.Cleaner")

class TextCleaner:
    """Cleans extracted legal text by removing noise while preserving structural elements."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.settings = self.config.cleaning_settings
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compiles regular expressions from configuration for performance."""
        self.header_footer_regexes = []
        patterns = self.settings.get("header_footer_patterns", [])
        for pattern in patterns:
            try:
                self.header_footer_regexes.append(re.compile(pattern))
            except re.error as e:
                logger.error("Invalid cleaning pattern '%s': %s", pattern, str(e))
                
        # Standard page number regex (line with just a number or "Page X", "X of Y")
        self.page_number_regex = re.compile(
            r"^\s*(?:-?\s*\d+\s*-?|page\s+\d+|\d+\s*/\s*\d+|\d+\s+of\s+\d+)\s*$", 
            re.IGNORECASE
        )
        
        # OCR artifact cleanup regexes
        self.ocr_hyphen_regex = re.compile(r"(\w+)-\n+(\w+)") # re-join split words at line ends

    def clean_text(self, text: str) -> str:
        """Applies configuration cleaning steps to the raw text content."""
        if not text:
            return ""

        # 1. Split into lines to clean line-by-line (headers/footers/page numbers)
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        
        remove_hf = self.settings.get("remove_headers_footers", True)
        remove_pn = self.settings.get("remove_page_numbers", True)

        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines during line filter (we will normalize spacing later)
            if not stripped:
                cleaned_lines.append("")
                continue

            # Skip page break marker used by PDF/DOCX parser
            if stripped == "--- PAGE BREAK ---":
                continue

            # Check page numbers
            if remove_pn and self.page_number_regex.match(stripped):
                logger.debug("Removing page number line: '%s'", stripped)
                continue

            # Check header/footers
            is_hf = False
            if remove_hf:
                for regex in self.header_footer_regexes:
                    if regex.match(stripped):
                        logger.debug("Removing header/footer line: '%s'", stripped)
                        is_hf = True
                        break
            if is_hf:
                continue

            # Keep valid line
            cleaned_lines.append(line)

        # Re-join filtered lines
        cleaned_text = "\n".join(cleaned_lines)

        # 2. Clean OCR artifacts
        if self.settings.get("strip_ocr_artifacts", True):
            # Re-join words broken by line endings (e.g. "con-\nstitution" -> "constitution")
            cleaned_text = self.ocr_hyphen_regex.sub(r"\1\2", cleaned_text)
            
            # Clean weird non-printable characters or weird symbol sequences, but preserve standard formatting
            # Keep standard ASCII, letters, numbers, punctuation, and newlines
            cleaned_text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u00FF\u2010-\u201d\u2022]", "", cleaned_text)

        # 3. Normalize whitespace
        if self.settings.get("normalize_whitespace", True):
            # Strip multiple blank lines (max 2 consecutive newlines)
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
            
            # Remove double spaces on a single line but keep indents
            # We preserve leading spaces since indentations represent structure in legal drafts
            def collapse_double_spaces(match):
                line = match.group(0)
                indent = len(line) - len(line.lstrip())
                return " " * indent + re.sub(r"[ \t]{2,}", " ", line.lstrip())
                
            lines_norm = [re.sub(r"^[ \t]*.*$", collapse_double_spaces, l) for l in cleaned_text.splitlines()]
            cleaned_text = "\n".join(lines_norm)
            
            # Final trim
            cleaned_text = cleaned_text.strip()

        return cleaned_text

    def clean_document(self, doc_id: str) -> bool:
        """Cleans a single document's processed text and writes it to the cleaned directory."""
        metadata = self.metadata_mgr.get_document_metadata(doc_id)
        if not metadata:
            logger.error("Cannot clean document %s: metadata not found.", doc_id)
            return False

        processed_path = metadata.get("processed_file_path")
        if not processed_path or not os.path.exists(processed_path):
            logger.error("Processed file not found for document %s: %s", doc_id, processed_path)
            return False

        logger.info("Cleaning document: %s (%s)", doc_id, metadata.get("original_filename"))

        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = self.clean_text(raw_text)

            cleaned_dir = self.config.cleaned_dir
            cleaned_file_path = os.path.normpath(os.path.join(cleaned_dir, f"{doc_id}.txt"))

            with open(cleaned_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            logger.info("Saved cleaned document to %s", cleaned_file_path)

            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "cleaned",
                "cleaned_file_path": cleaned_file_path,
                "error_log": None
            })
            return True

        except Exception as e:
            logger.error("Failed to clean document %s: %s", doc_id, str(e), exc_info=True)
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "failed",
                "error_log": f"Cleaning Error: {str(e)}"
            })
            return False

    def clean_all(self) -> Dict[str, int]:
        """Cleans all processed documents in the registry."""
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for doc_id, meta in list(self.metadata_mgr.registry.items()):
            # We clean files that are in 'processed' state
            # Or we can re-clean if already 'cleaned' / 'chunked'
            status = meta.get("processing_status")
            if status == "processed":
                success = self.clean_document(doc_id)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
            elif status in ["cleaned", "chunked"]:
                results["skipped"] += 1
            else:
                results["skipped"] += 1
                
        logger.info("Cleaning run completed. Results: %s", results)
        return results
