import os
import shutil
import tempfile
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional
from src.config import Config
from src.metadata import MetadataManager
from src.parsers import ParserFactory
from src.chunker import SemanticChunker
from src.retrieval.retrieval_service import RetrievalService
from src.services.exceptions import FileValidationError, ServiceError

logger = logging.getLogger("CrackLaw.Services.DocumentService")

class DocumentService:
    """Manages legal document uploads, metadata registrations, parsing, chunking, and indexing."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.supported_extensions = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}

    def upload_document(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Validates, parses, chunks, and indexes an uploaded document in the vector store."""
        # 1. Size check: 10MB limit
        if len(content) > 10 * 1024 * 1024:
            raise FileValidationError("File size exceeds the maximum limit of 10MB.")

        # 2. Extension check
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_extensions:
            raise FileValidationError(f"Unsupported file format '{ext}'. Allowed: {self.supported_extensions}")

        # 3. Write payload to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            with open(temp_path, "wb") as f:
                f.write(content)

            # 4. Parse content
            parser = ParserFactory.get_parser(temp_path)
            parsed_data = parser.parse(temp_path)
            extracted_text = parsed_data.get("text", "")
            
            if not extracted_text.strip():
                raise FileValidationError("Parsed document content is empty.")

            # Compute content-based document ID
            doc_id = "doc_" + hashlib.md5(extracted_text.encode("utf-8")).hexdigest()[:16]

            # 5. Move original file to the raw manual directory
            raw_uploads_dir = os.path.join(self.config.raw_dir, "manual_uploads")
            os.makedirs(raw_uploads_dir, exist_ok=True)
            raw_dest_path = os.path.normpath(os.path.join(raw_uploads_dir, f"{doc_id}{ext}"))
            shutil.copy2(temp_path, raw_dest_path)

            # Register in registry metadata file
            self.metadata_mgr.register_document(
                file_path=raw_dest_path,
                source="manual_upload",
                doc_type="manual_uploads"
            )

            # Save clean text
            cleaned_dir = os.path.join(self.config.cleaned_dir, "manual_uploads")
            os.makedirs(cleaned_dir, exist_ok=True)
            cleaned_dest_path = os.path.normpath(os.path.join(cleaned_dir, f"{doc_id}.txt"))
            
            with open(cleaned_dest_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)

            self.metadata_mgr.update_metadata(doc_id, {
                "cleaned_file_path": cleaned_dest_path,
                "processing_status": "cleaned"
            })

            # 6. Run Semantic Chunker
            chunker = SemanticChunker(self.config, self.metadata_mgr)
            chunk_success = chunker.chunk_document(doc_id)
            if not chunk_success:
                raise ServiceError(f"Semantic chunking pipeline failed for document {doc_id}")

            # 7. Index in Vector Store
            retrieval = RetrievalService(self.config)
            indexed_count = retrieval.index_document(doc_id)

            logger.info("Uploaded and indexed document: doc_id=%s, chunks=%d", doc_id, indexed_count)
            return {
                "document_id": doc_id,
                "filename": filename,
                "status": "indexed",
                "chunks_count": indexed_count,
                "char_length": len(extracted_text)
            }
        except Exception as e:
            if isinstance(e, FileValidationError):
                raise e
            logger.error("Failed to parse and index document upload: %s", str(e), exc_info=True)
            raise ServiceError(f"Document upload processing error: {e}") from e
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Extracts structures, sections, and tags from raw document text."""
        if not text.strip():
            raise FileValidationError("Document text cannot be empty.")
            
        try:
            # We can use the feature extractor and clean text helpers to analyze the text structure
            from src.models.feature_engineering import LegalFeatureExtractor
            from src.models.preprocessing import clean_text
            
            extractor = LegalFeatureExtractor()
            feats = extractor.extract_features_from_text(text)
            
            # Simple keyword matching for acts/sections
            acts = []
            sections = []
            
            # Regex for section matches
            for m in re.finditer(r"(?i)\bsection\s+(\d+[A-Z]?)\b", text):
                sections.append(f"Section {m.group(1)}")
            for m in re.finditer(r"(?i)\b(?:the\s+)?([A-Za-z\s]+Act,\s+\d{4})\b", text):
                acts.append(m.group(1).strip())
                
            return {
                "char_length": len(text),
                "word_count": len(text.split()),
                "legal_feature_scores": feats,
                "extracted_acts": list(set(acts)),
                "extracted_sections": list(set(sections))
            }
        except Exception as e:
            logger.error("Error in analyze_document: %s", str(e))
            raise ServiceError(f"Document analysis error: {e}") from e
