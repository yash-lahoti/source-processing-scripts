#!/usr/bin/env python3
"""
Advanced aggregation script with config file selection.
Allows you to specify a custom config file and data directory.
"""
import sys
import argparse
from pathlib import Path

# Add src to path so we can import the package
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from patient_aggregator import aggregate_patients


def main():
    """Main function with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate patient data from Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default data directory and config
  python run_aggregation_with_config.py
  
  # Specify custom data directory
  python run_aggregation_with_config.py --data-dir ./my_data
  
  # Specify custom config file
  python run_aggregation_with_config.py --config custom_config.yaml
  
  # Specify both
  python run_aggregation_with_config.py --data-dir ./my_data --config custom_config.yaml --output results.csv
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing Excel files (default: auto-detect)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file (default: config.yaml in project root)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="aggregated_patients.csv",
        help="Output CSV file path (default: aggregated_patients.csv)"
    )
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Error: Data directory not found: {data_dir}")
            return 1
    else:
        # Auto-detect
        project_root = Path(__file__).parent
        possible_dirs = [
            project_root / "data",
            project_root / "sample_data",
            project_root / "input_data",
            project_root,
        ]
        
        data_dir = None
        for possible_dir in possible_dirs:
            if possible_dir.exists() and list(possible_dir.glob("*.xlsx")):
                data_dir = possible_dir
                break
        
        if data_dir is None:
            print("❌ Error: Could not find data directory with Excel files.")
            print("Please specify --data-dir")
            return 1
    
    print(f"Data directory: {data_dir}")
    
    # Check config file
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Error: Config file not found: {config_path}")
            return 1
        print(f"Config file: {config_path}")
    else:
        config_path = project_root / "config.yaml"
        if config_path.exists():
            print(f"Using default config: {config_path}")
        else:
            print("⚠ Warning: Default config.yaml not found, using package defaults")
            config_path = None
    
    # Output file
    output_file = Path(args.output)
    print(f"Output file: {output_file}")
    print()
    
    # Run aggregation
    try:
        print("Starting aggregation...")
        aggregate_patients(
            input_dir=str(data_dir),
            output_file=str(output_file),
            config_path=str(config_path) if config_path else None
        )
        print(f"✓ Success! Results saved to {output_file}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

