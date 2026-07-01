import os
import logging
from typing import Dict, Any, Optional
from src.config import Config
from src.metadata import MetadataManager
from src.parsers import ParserFactory
from src.utils import scan_files_by_extension, get_relative_path

logger = logging.getLogger("CrackLaw.Ingestion")

class IngestionPipeline:
    """Automates scanning raw datasets, parsing contents, and writing extracted text."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)

    def get_category_from_path(self, file_path: str) -> str:
        """Determines the dataset category based on its raw directory subfolder."""
        rel_path = get_relative_path(file_path, self.config.raw_dir)
        parts = rel_path.split("/")
        if len(parts) > 1:
            return parts[0]
        return "miscellaneous"

    def ingest_file(self, file_path: str) -> bool:
        """Ingests a single file: parses, saves text, and updates metadata."""
        category = self.get_category_from_path(file_path)
        logger.info("Ingesting file: %s (Category: %s)", file_path, category)
        
        # 1. Register or get existing metadata
        try:
            metadata = self.metadata_mgr.register_document(
                file_path=file_path,
                source="manual_import" if "manual" in file_path else "external_source",
                doc_type=category
            )
            doc_id = metadata["id"]
        except Exception as e:
            logger.error("Failed to register document %s: %s", file_path, str(e))
            return False

        # If already successfully processed/cleaned/chunked, we can skip unless requested
        if metadata.get("processing_status") in ["processed", "cleaned", "chunked"]:
            logger.info("Document %s is already processed. Skipping.", doc_id)
            return True

        # 2. Dispatch to parser
        try:
            parser = ParserFactory.get_parser(file_path)
            parsed_data = parser.parse(file_path)
            extracted_text = parsed_data["text"]
            extracted_meta = parsed_data["metadata"]
            
            # Update title from parser if available
            if extracted_meta.get("title") and extracted_meta["title"] != os.path.splitext(os.path.basename(file_path))[0]:
                self.metadata_mgr.update_metadata(doc_id, {"document_title": extracted_meta["title"]})
                
        except Exception as e:
            logger.error("Error parsing file %s: %s", file_path, str(e), exc_info=True)
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "failed",
                "error_log": f"Parsing Error: {str(e)}"
            })
            return False

        # 3. Save extracted text to processed folder
        try:
            dest_category_dir = os.path.join(self.config.processed_dir, category)
            os.makedirs(dest_category_dir, exist_ok=True)
            processed_file_path = os.path.normpath(os.path.join(dest_category_dir, f"{doc_id}.txt"))
            
            with open(processed_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
                
            logger.info("Saved extracted text to %s", processed_file_path)
            
            # Update metadata
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "processed",
                "processed_file_path": processed_file_path,
                "error_log": None
            })
            return True
            
        except Exception as e:
            logger.error("Failed to save processed file for %s: %s", file_path, str(e))
            self.metadata_mgr.update_metadata(doc_id, {
                "processing_status": "failed",
                "error_log": f"Save Processed Error: {str(e)}"
            })
            return False

    def ingest_all(self) -> Dict[str, int]:
        """Scans raw directory and ingests all supported documents."""
        supported_formats = self.config.formats
        logger.info("Scanning for raw documents with extensions: %s", supported_formats)
        
        all_raw_files = scan_files_by_extension(self.config.raw_dir, set(supported_formats))
        logger.info("Found %d raw files to ingest.", len(all_raw_files))
        
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        for file_path in all_raw_files:
            # Skip metadata and registry files if they are in raw/ by accident
            if "dataset_registry" in file_path or file_path.endswith(".json") and "metadata" in file_path:
                continue
                
            # Perform ingestion
            success = self.ingest_file(file_path)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
                
        logger.info("Ingestion run completed. Results: %s", results)
        return results
