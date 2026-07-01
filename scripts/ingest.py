import os
import sys
import argparse
import logging

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.ingestion import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Ingestion CLI: Scan datasets/raw/ and parse documents to datasets/processed/."
    )
    parser.add_argument("--file", help="Ingest a single specific raw file instead of scanning the directory.")
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Ingest")
    logger.info("Initializing CrackLaw Ingestion CLI...")

    pipeline = IngestionPipeline(config)

    try:
        if args.file:
            if not os.path.exists(args.file):
                logger.error("Specified file not found: %s", args.file)
                print(f"Error: file not found at {args.file}", file=sys.stderr)
                sys.exit(1)
                
            logger.info("Running single file ingestion for %s", args.file)
            success = pipeline.ingest_file(args.file)
            if success:
                print(f"Successfully ingested file: {args.file}")
            else:
                print(f"Failed to ingest file: {args.file}. See logs for details.", file=sys.stderr)
                sys.exit(1)
        else:
            logger.info("Running complete raw/ directory scan and ingestion...")
            results = pipeline.ingest_all()
            print("\n==================================================")
            print("INGESTION RUN COMPLETED")
            print("==================================================")
            print(f"Parsed Successfully: {results['success']}")
            print(f"Failed to Parse:     {results['failed']}")
            print(f"Skipped (Already OK): {results['skipped']}")
            print("==================================================")
            if results['failed'] > 0:
                print("Note: Check logs/cracklaw.log for error descriptions.")

    except Exception as e:
        logger.error("Ingestion command failed: %s", str(e), exc_info=True)
        print(f"Critical Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
