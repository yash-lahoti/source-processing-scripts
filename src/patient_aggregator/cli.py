"""CLI entry point for patient aggregation."""
import argparse
from .aggregator import aggregate_patients


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Aggregate patient data from Excel files")
    parser.add_argument("--input-dir", required=True, help="Directory containing Excel files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--config", help="Path to config YAML file (default: config.yaml in project root)")
    args = parser.parse_args()
    aggregate_patients(args.input_dir, args.output, args.config)
    print(f"Aggregation complete. Output saved to {args.output}")


if __name__ == "__main__":
    main()

