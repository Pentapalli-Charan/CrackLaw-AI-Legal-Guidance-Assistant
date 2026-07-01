import os
import re
import json
import logging
from typing import Optional, Dict, List, Any, Tuple
from src.config import Config
from src.metadata import MetadataManager
from src.utils import estimate_word_count

logger = logging.getLogger("CrackLaw.Chunker")

class SemanticChunker:
    """Chunks legal documents hierarchically (Act -> Chapter -> Section -> Subsection)
    with fallback to paragraph-based chunking with overlap."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.settings = self.config.chunking_settings
        self._compile_regexes()

    def _compile_regexes(self) -> None:
        """Pre-compiles structural regexes for identifying hierarchical units."""
        patterns = self.settings.get("hierarchy_regexes", {})
        
        # We compile the patterns. If not configured, we use robust defaults.
        self.act_regex = re.compile(patterns.get("act", r"(?i)^\s*(?:THE\s+)?(?:[A-Za-z0-9_\s\-]+)?ACT,\s+\d{4}"))
        self.chapter_regex = re.compile(patterns.get("chapter", r"(?i)^\s*(?:CHAPTER|PART)\s+([IVXLCDM\d]+)"))
        
        # Captures "Section 12", "Sec. 12", "12. Section Name", etc.
        self.section_regex = re.compile(patterns.get("section", r"(?i)^\s*(?:SECTION|SEC\.)\s+(\d+[A-Za-z]?)|^\s*(\d+[A-Za-z]?)\.\s+[A-Z]"))
        self.subsection_regex = re.compile(patterns.get("subsection", r"^\s*\((\d+|[a-zA-Z]+)\)\s+"))

    def _detect_hierarchy(self, paragraph: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Scans a paragraph to see if it introduces a new Act, Chapter, Section, or Subsection."""
        stripped = paragraph.strip()
        if not stripped:
            return None, None, None, None

        act_match = self.act_regex.match(stripped)
        if act_match:
            return act_match.group(0).strip(), None, None, None

        chapter_match = self.chapter_regex.match(stripped)
        if chapter_match:
            return None, chapter_match.group(0).strip(), None, None

        section_match = self.section_regex.match(stripped)
        if section_match:
            sec_val = section_match.group(0).strip()
            # Trim trailing period if it is like "12. "
            if sec_val.endswith("."):
                sec_val = sec_val[:-1]
            return None, None, sec_val, None

        sub_match = self.subsection_regex.match(stripped)
        if sub_match:
            return None, None, None, sub_match.group(0).strip()

        return None, None, None, None

    def chunk_hierarchically(self, text: str) -> List[Dict[str, Any]]:
        """Attempts to chunk text by legal structure."""
        paragraphs = text.split("\n\n")
        chunks: List[Dict[str, Any]] = []
        
        current_act = None
        current_chapter = None
        current_section = None
        current_subsection = None
        
        buffer_text = []
        buffer_words = 0
        max_words = self.settings.get("max_chunk_words", 500)
        
        chunk_idx = 0
        
        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            # Detect hierarchy transitions
            act, chap, sec, sub = self._detect_hierarchy(para_stripped)
            
            # If any structural change is detected, we emit the buffer if it is non-empty
            if (act or chap or sec) and buffer_text:
                chunk_text = "\n\n".join(buffer_text)
                chunks.append({
                    "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                    "text": chunk_text,
                    "act": current_act,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "word_count": buffer_words
                })
                chunk_idx += 1
                buffer_text = []
                buffer_words = 0
                
            # Update active hierarchy state
            if act:
                current_act = act
                current_chapter = None
                current_section = None
                current_subsection = None
            if chap:
                current_chapter = chap
                current_section = None
                current_subsection = None
            if sec:
                current_section = sec
                current_subsection = None
            if sub:
                current_subsection = sub
                
            # Add paragraph to buffer
            para_words = estimate_word_count(para_stripped)
            
            # If a single paragraph is extremely large, we must split it even within a section
            if buffer_words + para_words > max_words and buffer_text:
                # Emit current buffer
                chunk_text = "\n\n".join(buffer_text)
                chunks.append({
                    "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                    "text": chunk_text,
                    "act": current_act,
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "word_count": buffer_words
                })
                chunk_idx += 1
                buffer_text = [para_stripped]
                buffer_words = para_words
            else:
                buffer_text.append(para_stripped)
                buffer_words += para_words

        # Emit trailing buffer
        if buffer_text:
            chunk_text = "\n\n".join(buffer_text)
            chunks.append({
                "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                "text": chunk_text,
                "act": current_act,
                "chapter": current_chapter,
                "section": current_section,
                "subsection": current_subsection,
                "word_count": buffer_words
            })

        return chunks

    def chunk_by_paragraphs_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Paragraph-based chunking with configurable limits and overlap (used if no hierarchy found)."""
        paragraphs = text.split("\n\n")
        chunks: List[Dict[str, Any]] = []
        
        max_words = self.settings.get("max_chunk_words", 500)
        overlap_words = self.settings.get("chunk_overlap_words", 50)
        
        current_words_list: List[str] = []
        chunk_idx = 0

        for para in paragraphs:
            para_stripped = para.strip()
            if not para_stripped:
                continue

            para_words = para_stripped.split()
            
            # If adding this paragraph exceeds max limit
            if len(current_words_list) + len(para_words) > max_words:
                if current_words_list:
                    # Emit current chunk
                    chunk_text = " ".join(current_words_list)
                    chunks.append({
                        "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                        "text": chunk_text,
                        "act": None,
                        "chapter": None,
                        "section": None,
                        "subsection": None,
                        "word_count": len(current_words_list)
                    })
                    chunk_idx += 1
                    
                    # Apply overlap: take the last N words from the emitted chunk
                    overlap = current_words_list[-overlap_words:] if len(current_words_list) > overlap_words else current_words_list
                    current_words_list = list(overlap) + para_words
                else:
                    # Paragraph itself is larger than max_words, chunk it directly
                    # Let's split this massive paragraph into word slices
                    i = 0
                    while i < len(para_words):
                        slice_words = para_words[i:i + max_words]
                        chunk_text = " ".join(slice_words)
                        chunks.append({
                            "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                            "text": chunk_text,
                            "act": None,
                            "chapter": None,
                            "section": None,
                            "subsection": None,
                            "word_count": len(slice_words)
                        })
                        chunk_idx += 1
                        i += (max_words - overlap_words) if max_words > overlap_words else max_words
                    current_words_list = []
            else:
                current_words_list.extend(para_words)

        # Emit trailing chunk
        if current_words_list:
            chunk_text = " ".join(current_words_list)
            chunks.append({
                "chunk_id_suffix": f"chunk_{chunk_idx:03d}",
                "text": chunk_text,
                "act": None,
                "chapter": None,
                "section": None,
                "subsection": None,
                "word_count": len(current_words_list)
            })
            
        return chunks

    def chunk_document(self, doc_id: str) -> bool:
        """Loads a cleaned document, chunks it, and saves the chunks JSON file."""
        metadata = self.metadata_mgr.get_document_metadata(doc_id)
        if not metadata:
            logger.error("Cannot chunk document %s: metadata not found.", doc_id)
            return False

        cleaned_path = metadata.get("cleaned_file_path")
        if not cleaned_path or not os.path.exists(cleaned_path):
            logger.error("Cleaned file not found for document %s: %s", doc_id, cleaned_path)
            return False

        logger.info("Chunking document: %s (%s)", doc_id, metadata.get("original_filename"))

        try:
            with open(cleaned_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 1. Attempt hierarchical chunking
            chunks_data = []
            if self.settings.get("semantic_split", True):
                chunks_data = self.chunk_hierarchically(text)
                
                # Check if we successfully parsed hierarchy (i.e. did we actually match acts/chapters/sections?)
                has_hierarchy = any(c["act"] or c["chapter"] or c["section"] for c in chunks_data)
                if not has_hierarchy:
                    logger.debug("No legal hierarchy matched. Falling back to paragraph-based chunking with overlap.")
                    chunks_data = self.chunk_by_paragraphs_fallback(text)
            else:
                chunks_data = self.chunk_by_paragraphs_fallback(text)

            # 2. Enrich chunks with document metadata
            final_chunks = []
            for chunk in chunks_data:
                chunk_id = f"{doc_id}_{chunk['chunk_id_suffix']}"
                final_chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "source": metadata.get("source"),
                    "act": chunk["act"],
                    "chapter": chunk["chapter"],
                    "section": chunk["section"],
                    "subsection": chunk["subsection"],
                    "language": metadata.get("language", "en"),
                    "text": chunk["text"],
                    "word_count": chunk["word_count"]
                })

            # 3. Save chunks JSON to chunks directory
            chunks_dir = self.config.chunks_dir
            chunks_file_path = os.path.normpath(os.path.join(chunks_dir, f"{doc_id}_chunks.json"))
            
            with open(chunks_file_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, indent=2)

            logger.info("Saved %d chunks to %s", len(final_chunks), chunks_file_path)

            # Update document metadata status
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "chunked",
                "chunks_file_path": chunks_file_path,
                "chunk_count": len(final_chunks),
                "error_log": None
            })
            return True

        except Exception as e:
            logger.error("Failed to chunk document %s: %s", doc_id, str(e), exc_info=True)
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "failed",
                "error_log": f"Chunking Error: {str(e)}"
            })
            return False

    def chunk_all(self) -> Dict[str, int]:
        """Chunks all cleaned documents in the registry."""
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for doc_id, meta in list(self.metadata_mgr.registry.items()):
            # We chunk files that are in 'cleaned' state
            status = meta.get("processing_status")
            if status == "cleaned":
                success = self.chunk_document(doc_id)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
            elif status == "chunked":
                results["skipped"] += 1
            else:
                results["skipped"] += 1
                
        logger.info("Chunking run completed. Results: %s", results)
        return results
