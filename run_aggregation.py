#!/usr/bin/env python3
"""
Standalone script to aggregate patient data from Excel files.
Scans the data directory and automatically processes available files.
Can be run directly from an IDE or command line.
"""
import sys
from pathlib import Path

# Add src to path so we can import the package
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from patient_aggregator import aggregate_patients
from patient_aggregator.visualizer import visualize_aggregated_data


def find_data_directory():
    """Find the data directory, checking common locations."""
    project_root = Path(__file__).parent
    
    # Check common data directory names
    possible_dirs = [
        project_root / "data",
        project_root / "sample_data",
        project_root / "input_data",
        project_root,  # Current directory as fallback
    ]
    
    for data_dir in possible_dirs:
        if data_dir.exists() and data_dir.is_dir():
            # Check if it contains Excel or CSV files
            data_files = list(data_dir.glob("*.xlsx"))
            data_files.extend(data_dir.glob("*.xls"))
            data_files.extend(data_dir.glob("*.csv"))
            if data_files:
                return data_dir
    
    return None


def scan_data_directory(data_dir):
    """Scan data directory and report what files are found."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return None, []
    
    # Look for Excel and CSV files
    data_files = list(data_path.glob("*.xlsx"))
    data_files.extend(data_path.glob("*.xls"))
    data_files.extend(data_path.glob("*.csv"))
    
    # Check for expected files (with various extensions)
    # Match base names (without extension)
    expected_basenames = ["cohort", "iop", "diagnosis", "enc"]
    found_files = {}
    
    for file in data_files:
        filename_base = file.stem.lower()  # Get filename without extension (case-insensitive)
        for expected in expected_basenames:
            if expected.lower() == filename_base:  # Exact match on base name
                # Use the expected name with .xlsx extension as key for display
                key = f"{expected}.xlsx"
                found_files[key] = file
                break
    
    return data_path, found_files


def main():
    """Main function to run aggregation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate patient data and generate visualizations")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating visualizations")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Patient Data Aggregation Script")
    print("=" * 60)
    print()
    
    # Find data directory
    data_dir = find_data_directory()
    
    if data_dir is None:
        print("❌ Error: Could not find data directory with Excel or CSV files.")
        print()
        print("Please ensure one of these directories exists with data files (.xlsx, .xls, or .csv):")
        print("  - ./data/")
        print("  - ./sample_data/")
        print("  - ./input_data/")
        print("  - ./ (current directory)")
        return 1
    
    print(f"✓ Found data directory: {data_dir}")
    print()
    
    # Scan for files
    data_path, found_files = scan_data_directory(data_dir)
    
    if not found_files:
        print(f"❌ Error: No data files (.xlsx, .xls, or .csv) found in {data_dir}")
        print(f"   Please ensure files like 'cohort', 'iop', 'diagnosis', or 'enc' exist in this directory")
        return 1
    
    print("Found data files:")
    for expected_file in ["cohort.xlsx", "iop.xlsx", "diagnosis.xlsx", "enc.xlsx"]:
        if expected_file in found_files:
            actual_file = found_files[expected_file]
            print(f"  ✓ {expected_file} (found as {actual_file.name})")
        else:
            print(f"  ⚠ {expected_file} (not found - will be skipped)")
    print()
    
    # Set output file
    output_file = Path(__file__).parent / "aggregated_patients.csv"
    
    print(f"Output file: {output_file}")
    print()
    print("Starting aggregation...")
    print("-" * 60)
    
    try:
        # Run aggregation
        aggregate_patients(
            input_dir=str(data_path),
            output_file=str(output_file)
        )
        
        print("-" * 60)
        print("✓ Aggregation complete!")
        print()
        print(f"Results saved to: {output_file}")
        
        # Show summary
        try:
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"✓ Total patients aggregated: {len(df)}")
            print(f"✓ Columns: {len(df.columns)}")
            print()
            print("Columns:", ", ".join(df.columns.tolist()))
        except Exception as e:
            print(f"Note: Could not read output file for summary: {e}")
        
        # Generate visualizations
        if not args.no_plots:
            try:
                visualize_aggregated_data(str(output_file))
            except Exception as e:
                print(f"\n⚠ Warning: Could not generate visualizations: {e}")
                print("Aggregation completed successfully, but visualizations were skipped.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found: {e}")
        print("Please ensure all required data files (Excel or CSV) are in the data directory.")
        return 1
    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

