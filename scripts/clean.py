import sys
import os
import argparse
import logging

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.cleaner import TextCleaner

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Text Cleaning CLI: Sanitize processed files and output to datasets/cleaned/."
    )
    parser.add_argument("--doc-id", help="Clean a single specific registered document by its ID.")
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Clean")
    logger.info("Initializing CrackLaw Cleaner CLI...")

    cleaner = TextCleaner(config)

    try:
        if args.doc_id:
            logger.info("Running cleaning for document: %s", args.doc_id)
            success = cleaner.clean_document(args.doc_id)
            if success:
                print(f"Successfully cleaned document: {args.doc_id}")
            else:
                print(f"Failed to clean document: {args.doc_id}. See logs for details.", file=sys.stderr)
                sys.exit(1)
        else:
            logger.info("Running directory-wide cleaning...")
            results = cleaner.clean_all()
            print("\n==================================================")
            print("TEXT CLEANING RUN COMPLETED")
            print("==================================================")
            print(f"Cleaned Successfully: {results['success']}")
            print(f"Failed to Clean:      {results['failed']}")
            print(f"Skipped (Already OK):  {results['skipped']}")
            print("==================================================")
            if results['failed'] > 0:
                print("Note: Check logs/cracklaw.log for error descriptions.")

    except Exception as e:
        logger.error("Cleaning command failed: %s", str(e), exc_info=True)
        print(f"Critical Error during cleaning: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
