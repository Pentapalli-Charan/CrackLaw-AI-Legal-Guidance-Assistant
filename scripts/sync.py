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
        description="CrackLaw Dataset Synchronization CLI: Download, verify, and ingest pre-configured datasets."
    )
    parser.add_argument("--dataset", help="Synchronize a single specific dataset by its name.")
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Sync")
    logger.info("Initializing dataset sync CLI...")

    sync_service = SyncService(config)

    try:
        if args.dataset:
            print(f"Synchronizing specific dataset: {args.dataset} ...")
            ka_settings = config.get("knowledge_acquisition", {})
            datasets_cfg = ka_settings.get("datasets", [])
            target_cfg = None
            for d in datasets_cfg:
                if d.get("name") == args.dataset:
                    target_cfg = d
                    break
            
            if not target_cfg:
                print(f"Error: Dataset '{args.dataset}' not found in configuration.", file=sys.stderr)
                sys.exit(1)
                
            success = sync_service.sync_dataset(target_cfg)
            print("\n==================================================")
            print("           DATASET SYNC COMPLETED                 ")
            print("==================================================")
            print(f"Dataset Name:  {args.dataset}")
            print(f"Sync Outcome:  {'SUCCESS' if success else 'FAILED'}")
            print("==================================================")
            if not success:
                sys.exit(1)
        else:
            print("Synchronizing all configured datasets...")
            results = sync_service.sync_all()
            print("\n==================================================")
            print("           GLOBAL SYNC COMPLETED                  ")
            print("==================================================")
            print(f"Datasets Succeeded:  {results['success']}")
            print(f"Datasets Failed:     {results['failed']}")
            print("==================================================")
            if results["failed"] > 0:
                print("Note: Check logs/cracklaw.log for error descriptions.")
                
    except Exception as e:
        logger.error("Sync execution failed: %s", str(e), exc_info=True)
        print(f"Critical error during synchronization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
