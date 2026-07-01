import os
import sys
import argparse
import logging

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.downloader import DownloadManager

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Dataset Downloader: Download files from URLs, Hugging Face, or Kaggle, and copy manual files."
    )
    
    # Destination Category
    parser.add_argument(
        "--category",
        choices=[
            "laws", "judgments", "contracts", "legal_qa", "legal_nlp",
            "government_notifications", "regulations", "miscellaneous"
        ],
        required=True,
        help="Subdirectory target under datasets/raw/ (e.g., laws, judgments, contracts)."
    )

    # Download source arguments (Mutually exclusive group)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Direct HTTP/HTTPS URL of the file to download.")
    source_group.add_argument("--hf", help="Hugging Face repo ID (e.g., owner/repo).")
    source_group.add_argument("--kaggle", help="Kaggle dataset identifier slug (e.g., owner/dataset).")
    source_group.add_argument("--manual", help="Path to a local file to manually import and register.")

    # Additional options
    parser.add_argument("--filename", help="Custom name for the downloaded file (URLs only).")
    parser.add_argument("--hf-file", help="Filename of the file inside the Hugging Face repo.")
    parser.add_argument("--kaggle-file", help="Filename of the file inside the Kaggle dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite file if it already exists.")
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")

    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Download")
    logger.info("Initializing CrackLaw Downloader CLI...")

    manager = DownloadManager(config)

    try:
        if args.url:
            logger.info("Triggering URL download: %s", args.url)
            final_path = manager.download_from_url(
                url=args.url,
                dest_category=args.category,
                filename=args.filename,
                overwrite=args.overwrite
            )
            print(f"Successfully processed URL download. File is located at: {final_path}")
            
        elif args.hf:
            if not args.hf_file:
                logger.error("Hugging Face download requires --hf-file parameter.")
                parser.error("Hugging Face download requires --hf-file parameter.")
                
            logger.info("Triggering Hugging Face download from repo %s: %s", args.hf, args.hf_file)
            final_path = manager.download_from_hf(
                repo_id=args.hf,
                filename=args.hf_file,
                dest_category=args.category
            )
            print(f"Successfully processed Hugging Face download. File is located at: {final_path}")
            
        elif args.kaggle:
            if not args.kaggle_file:
                logger.error("Kaggle download requires --kaggle-file parameter.")
                parser.error("Kaggle download requires --kaggle-file parameter.")
                
            logger.info("Triggering Kaggle download from dataset %s: %s", args.kaggle, args.kaggle_file)
            final_path = manager.download_from_kaggle(
                dataset_slug=args.kaggle,
                filename=args.kaggle_file,
                dest_category=args.category
            )
            print(f"Successfully processed Kaggle download. File is located at: {final_path}")
            
        elif args.manual:
            logger.info("Triggering manual file import: %s", args.manual)
            final_path = manager.add_manual_file(
                src_path=args.manual,
                dest_category=args.category,
                overwrite=args.overwrite
            )
            print(f"Successfully imported manual file. File is copied to: {final_path}")

    except Exception as e:
        logger.error("Operation failed: %s", str(e), exc_info=True)
        print(f"Error executing download/import: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
