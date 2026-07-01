import sys
import os
import argparse
import logging

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.sync_service import SyncService

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Local Folder Import CLI: Scan, verify, classify, and import local legal documents."
    )
    parser.add_argument("--path", required=True, help="Path to the local folder containing documents to import.")
    parser.add_argument("--category", default="miscellaneous", help="Default target category if classifier fails to categorize.")
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Import")
    logger.info("Initializing local import CLI...")

    sync_service = SyncService(config)

    try:
        print(f"Scanning and importing from folder: {args.path} ...\n")
        results = sync_service.import_local_folder(args.path, args.category)
        
        print("==================================================")
        print("           IMPORT RUN COMPLETED                   ")
        print("==================================================")
        print(f"Total Files Scanned:     {results['total_files']}")
        print(f"Successfully Imported:   {results['imported']}")
        print(f"Skipped (Duplicates):    {results['skipped_duplicates']}")
        print(f"Failed Verification:     {results['failed_verification']}")
        print("==================================================")
        
        if results["failed_verification"] > 0:
            print("\nVerification Failures detail:")
            for item in results["details"]:
                if item["status"] == "failed":
                    print(f"  - {item['file']}: {item['reason']}")
            print("==================================================")
            
    except Exception as e:
        logger.error("Import command failed: %s", str(e), exc_info=True)
        print(f"Error during import execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
