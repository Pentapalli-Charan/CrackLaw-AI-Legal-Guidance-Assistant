import sys
import os
import argparse
import json

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.registry import KnowledgeRegistry

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Registry CLI: Inspect registered datasets and synchronization metadata."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    parser.add_argument("--json", action="store_true", help="Print registry output in raw JSON format.")
    
    args = parser.parse_args()

    # Initialize configuration
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    registry_mgr = KnowledgeRegistry(config)
    datasets = registry_mgr.list_datasets()

    if args.json:
        print(json.dumps(registry_mgr.registry, indent=2))
        return

    print("==================================================")
    print("           CRACKLAW KNOWLEDGE REGISTRY            ")
    print("==================================================")
    if not datasets:
        print("No datasets registered yet. Run sync.py to register configured sources.")
    else:
        for idx, d in enumerate(datasets):
            print(f"{idx + 1}. Dataset Name:  {d['name']}")
            print(f"   Source Type:   {d['source_type']}")
            print(f"   Description:   {d.get('description', '')}")
            print(f"   Version:       {d['version']}")
            print(f"   License:       {d['license']}")
            print(f"   Languages:     {', '.join(d['supported_languages'])}")
            print(f"   Doc Types:     {', '.join(d['document_types'])}")
            print(f"   Sync Status:   {d['download_status'].upper()}")
            print(f"   Proc Status:   {d['processing_status'].upper()}")
            print(f"   Last Updated:  {d['last_updated'] or 'Never'}")
            print("-" * 50)
    print("==================================================")

if __name__ == "__main__":
    main()
