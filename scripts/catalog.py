import sys
import os
import argparse
import json

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.catalog import KnowledgeCatalog

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Catalog CLI: Compile and display the knowledge base catalog report."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    parser.add_argument("--json", action="store_true", help="Print catalog report in raw JSON format.")
    
    args = parser.parse_args()

    # Initialize configuration
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    catalog_mgr = KnowledgeCatalog(config)
    
    try:
        report = catalog_mgr.generate_catalog()
    except Exception as e:
        print(f"Error generating catalog report: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    sum_info = report["summary"]
    
    print("==================================================")
    print("           CRACKLAW KNOWLEDGE CATALOG             ")
    print("==================================================")
    print(f"Last Synchronization: {report['last_synchronization'] or 'Never'}")
    print(f"Registered Source Datasets: {report['knowledge_registry_datasets']}")
    print("-" * 50)
    print("DATABASE WORKLOAD SUMMARY:")
    print(f"  - Total Ingested Documents:  {sum_info['total_documents']}")
    print(f"  - Total Generated Chunks:    {sum_info['total_chunks']}")
    print(f"  - Duplicate Files Skipped:   {sum_info['duplicate_count']}")
    print(f"  - Failed Sync Downloads:     {sum_info['failed_downloads']}")
    print(f"  - Total Disk Footprint:      {sum_info['total_storage_size_mb']} MB")
    print(f"     - Raw Files:              {sum_info['raw_storage_mb']} MB")
    print(f"     - Processed Text:         {sum_info['processed_storage_mb']} MB")
    print(f"     - Cleaned Text:           {sum_info['cleaned_storage_mb']} MB")
    print(f"     - Chunks JSON:            {sum_info['chunks_storage_mb']} MB")
    print("-" * 50)
    print("DOCUMENT TYPES FREQUENCY:")
    for doc_type, count in report["document_types"].items():
        print(f"  - {doc_type:<20}: {count}")
    if not report["document_types"]:
        print("  - No documents registered.")
    print("-" * 50)
    print("PROCESSING STATE PROGRESS:")
    for status, count in report["processing_progress"].items():
        print(f"  - {status:<20}: {count}")
    if not report["processing_progress"]:
        print("  - No processing state records.")
    print("-" * 50)
    print("DATA SOURCES INTEGRATED:")
    for source, count in report["sources"].items():
        # Truncate source name if too long
        src_label = source if len(source) <= 25 else source[:22] + "..."
        print(f"  - {src_label:<25}: {count}")
    if not report["sources"]:
        print("  - No sources indexed.")
    print("==================================================")

if __name__ == "__main__":
    main()
