import sys
import os
import argparse

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import Config
from src.adapters import AdapterFactory

def main():
    parser = argparse.ArgumentParser(
        description="CrackLaw Data Sources CLI: List available source adapters and their configuration status."
    )
    parser.add_argument("--config-file", help="Path to a custom configuration JSON file.")
    args = parser.parse_args()

    # Initialize configuration
    try:
        config = Config(config_path=args.config_file) if args.config_file else Config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    ka_config = config.get("knowledge_acquisition", {})
    enabled_adapters = ka_config.get("enabled_adapters", [])

    print("==================================================")
    print("           CRACKLAW DATA ADAPTERS                 ")
    print("==================================================")
    
    # Pre-configured adapters in factory
    all_adapters = ["url", "local_folder", "huggingface", "kaggle", "indiacode", "supreme_court"]
    
    for adapter in all_adapters:
        status = "ENABLED" if adapter in enabled_adapters else "DISABLED"
        
        # Details about each adapter type
        details = ""
        if adapter == "url":
            details = "Downloads direct web files; supports Range resuming & hash validation"
        elif adapter == "local_folder":
            details = "Walks local directories; imports & classifies new documents"
        elif adapter == "huggingface":
            details = "Fetches datasets using HF hub client; serializes records to JSON"
        elif adapter == "kaggle":
            details = "Pulls Kaggle archives via API; extracts ZIP files dynamically"
        elif adapter == "indiacode":
            details = "Official legislative crawler; scrapes and parses Acts & Bills"
        elif adapter == "supreme_court":
            details = "Judicial judgment crawler; queries and pulls appellate orders"
            
        print(f"[{status:<8}] {adapter:<15}")
        print(f"           Description: {details}")
        print("-" * 50)
    print("==================================================")

if __name__ == "__main__":
    main()
