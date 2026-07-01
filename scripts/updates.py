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
        description="CrackLaw Updates Checker CLI: Scan for updates on configured datasets."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    args = parser.parse_args()

    # Initialize configuration and logger
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    logger = logging.getLogger("CrackLaw.CLI.Updates")
    logger.info("Initializing updates checker CLI...")

    sync_service = SyncService(config)

    try:
        print("Scanning configured sources for updates...")
        updates = sync_service.check_for_updates()
        
        print("\n==================================================")
        print("           UPDATE STATUS REPORT                   ")
        print("==================================================")
        if not updates:
            print("✓ Up to date: No new updates or datasets detected.")
        else:
            print(f"Found {len(updates)} pending updates or new datasets:")
            for item in updates:
                print(f"  - Name:        {item.get('name')}")
                print(f"    Source Type: {item.get('source_type')}")
                print(f"    Description: {item.get('description', '')}")
                print("    " + "-" * 30)
        print("==================================================")

    except Exception as e:
        logger.error("Updates check failed: %s", str(e), exc_info=True)
        print(f"Error checking updates: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
