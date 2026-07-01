import sys
import os
import argparse
import logging
from typing import Dict, Any, List

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.metadata import MetadataManager
from src.utils import calculate_checksum

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Verification CLI: Verify dataset integrity and audit processing states."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Verify")
    logger.info("Initializing CrackLaw Verification CLI...")

    metadata_mgr = MetadataManager(config)
    
    total_docs = len(metadata_mgr.registry)
    failures: List[Dict[str, Any]] = []

    print(f"Auditing {total_docs} registered documents...\n")

    for doc_id, reg_meta in metadata_mgr.registry.items():
        doc_meta = metadata_mgr.get_document_metadata(doc_id)
        if not doc_meta:
            failures.append({
                "id": doc_id,
                "file": reg_meta.get("original_filename"),
                "reason": "Missing individual metadata JSON file"
            })
            continue

        raw_file_path = doc_meta.get("file_path")
        
        # 1. Verify raw file existence
        if not raw_file_path or not os.path.exists(raw_file_path):
            failures.append({
                "id": doc_id,
                "file": doc_meta.get("original_filename"),
                "reason": f"Raw file not found at: {raw_file_path}"
            })
            continue

        # 2. Verify raw file checksum
        try:
            current_checksum = calculate_checksum(raw_file_path)
            expected_checksum = doc_meta.get("checksum")
            if current_checksum != expected_checksum:
                failures.append({
                    "id": doc_id,
                    "file": doc_meta.get("original_filename"),
                    "reason": f"Checksum mismatch. Expected: {expected_checksum[:16]}..., Actual: {current_checksum[:16]}..."
                })
                continue
        except Exception as e:
            failures.append({
                "id": doc_id,
                "file": doc_meta.get("original_filename"),
                "reason": f"Failed to compute checksum: {str(e)}"
            })
            continue

        # 3. Check status consistency and file paths
        status = doc_meta.get("processing_status")
        
        # Processed check
        if status in ["processed", "cleaned", "chunked"]:
            proc_path = doc_meta.get("processed_file_path")
            if not proc_path or not os.path.exists(proc_path):
                failures.append({
                    "id": doc_id,
                    "file": doc_meta.get("original_filename"),
                    "reason": f"Status is '{status}' but processed text file not found at: {proc_path}"
                })
                continue
                
        # Cleaned check
        if status in ["cleaned", "chunked"]:
            clean_path = doc_meta.get("cleaned_file_path")
            if not clean_path or not os.path.exists(clean_path):
                failures.append({
                    "id": doc_id,
                    "file": doc_meta.get("original_filename"),
                    "reason": f"Status is '{status}' but cleaned text file not found at: {clean_path}"
                })
                continue

        # Chunked check
        if status == "chunked":
            chunk_path = doc_meta.get("chunks_file_path")
            if not chunk_path or not os.path.exists(chunk_path):
                failures.append({
                    "id": doc_id,
                    "file": doc_meta.get("original_filename"),
                    "reason": f"Status is 'chunked' but chunks JSON file not found at: {chunk_path}"
                })
                continue

    # Final summary report
    print("==================================================")
    print("           VERIFICATION RUN COMPLETED             ")
    print("==================================================")
    if not failures:
        print("SUCCESS: All registered documents passed integrity and state audits.")
        print(f"Audited: {total_docs} files.")
        print("==================================================")
        sys.exit(0)
    else:
        print(f"FAILED: Found {len(failures)} verification warnings or errors:")
        for idx, fail in enumerate(failures):
            print(f"\n{idx + 1}. Doc ID: {fail['id']}")
            print(f"   File:   {fail['file']}")
            print(f"   Error:  {fail['reason']}")
        print("\n==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
