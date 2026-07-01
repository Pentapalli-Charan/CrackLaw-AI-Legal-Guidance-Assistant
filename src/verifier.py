import os
import json
import logging
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
from src.config import Config
from src.metadata import MetadataManager
from src.utils import calculate_checksum

logger = logging.getLogger("CrackLaw.Verifier")

class KnowledgeVerifier:
    """Verifies file integrity, structural corruption, and duplicates before ingestion."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)

    def is_file_corrupt(self, file_path: str) -> List[str]:
        """Tries to read or parse the file depending on its extension to detect corruption."""
        errors = []
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        size = os.path.getsize(file_path)
        
        # 1. Size sanity check
        if size == 0:
            errors.append("File is empty (0 bytes).")
            return errors

        # 2. Format parsing check
        try:
            if ext == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                # Try reading page count or metadata
                _ = len(reader.pages)
                
            elif ext == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    json.load(f)
                    
            elif ext == "xml":
                ET.parse(file_path)
                
            elif ext == "docx":
                from docx import Document
                Document(file_path)
                
            elif ext == "csv":
                import csv
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    # Check first few rows
                    for _, _ in zip(range(5), reader):
                        pass
                        
            elif ext in ["html", "htm"]:
                from bs4 import BeautifulSoup
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    # Check if bs4 could build a basic tree (soup is never None, but check element exists)
                    if not soup.html and not soup.body:
                        # Warning only, some fragments have no html/body tag
                        pass
        except Exception as e:
            errors.append(f"Format parsing error for '{ext}': {str(e)}")
            
        return errors

    def verify_file(self, file_path: str) -> Dict[str, Any]:
        """Runs validation checks (checksum, duplicate, corruption, completeness) on a file."""
        filename = os.path.basename(file_path)
        logger.info("Verifying integrity for file: %s", filename)
        
        report = {
            "file_path": file_path,
            "filename": filename,
            "is_valid": True,
            "checksum": None,
            "duplicate_doc_id": None,
            "errors": []
        }

        # 1. Verify existence
        if not os.path.exists(file_path):
            report["is_valid"] = False
            report["errors"].append("File does not exist.")
            return report

        # 2. Check incomplete download (e.g., matching .part file)
        part_file = file_path + ".part"
        if os.path.exists(part_file):
            report["is_valid"] = False
            report["errors"].append("Incomplete download: `.part` file exists.")

        # 3. Calculate checksum
        try:
            checksum = calculate_checksum(file_path)
            report["checksum"] = checksum
        except Exception as e:
            report["is_valid"] = False
            report["errors"].append(f"Failed to calculate checksum: {str(e)}")
            return report

        # 4. Detect duplicate
        duplicate_doc_id = self.metadata_mgr.check_duplicate(file_path)
        if duplicate_doc_id:
            report["duplicate_doc_id"] = duplicate_doc_id
            report["is_valid"] = False
            report["errors"].append(f"Duplicate document of registered ID: {duplicate_doc_id}")

        # 5. Check corruption
        corruption_errors = self.is_file_corrupt(file_path)
        if corruption_errors:
            report["is_valid"] = False
            report["errors"].extend(corruption_errors)

        return report

    def generate_report(self, file_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarizes verification statistics over a collection of file reports."""
        total = len(file_reports)
        valid = 0
        duplicates = 0
        corrupt = 0
        failures = []

        for r in file_reports:
            if r["is_valid"]:
                valid += 1
            else:
                failures.append(r)
                if r["duplicate_doc_id"]:
                    duplicates += 1
                else:
                    corrupt += 1

        return {
            "total_files_verified": total,
            "valid_files": valid,
            "duplicate_files": duplicates,
            "corrupt_files": corrupt,
            "failures": failures
        }
