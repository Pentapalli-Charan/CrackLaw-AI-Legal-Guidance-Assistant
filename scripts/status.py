import sys
import os
import argparse
import json

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.metadata import MetadataManager

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Status CLI: Display dataset statistics, size, and processing status."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    parser.add_argument("--json", action="store_true", help="Output stats in raw JSON format.")
    
    args = parser.parse_args()

    # Initialize configuration
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    metadata_mgr = MetadataManager(config)
    stats = metadata_mgr.generate_statistics()

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    print("==================================================")
    print("           CRACKLAW DATASET METRICS               ")
    print("==================================================")
    print(f"Total Registered Documents: {stats['total_documents']}")
    print(f"Total Chunks Generated:     {stats['total_chunks']}")
    print(f"Total Disk Space Utilized:  {stats['total_size_mb']} MB")
    print("--------------------------------------------------")
    print("PROCESSING STATUS DISTRIBUTION:")
    for status, count in stats['status_distribution'].items():
        print(f"  - {status:<15}: {count}")
    if not stats['status_distribution']:
        print("  - No files processed yet.")
        
    print("--------------------------------------------------")
    print("DOCUMENT CATEGORIES:")
    for category, count in stats['document_types'].items():
        print(f"  - {category:<15}: {count}")
    if not stats['document_types']:
        print("  - No categories registered.")
        
    print("--------------------------------------------------")
    print("DATASET SOURCES:")
    for source, count in stats['sources'].items():
        # Truncate source if too long
        src_label = source if len(source) <= 25 else source[:22] + "..."
        print(f"  - {src_label:<25}: {count}")
    if not stats['sources']:
        print("  - No sources registered.")
    print("==================================================")

if __name__ == "__main__":
    main()
