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
            # Check if it contains Excel files
            excel_files = list(data_dir.glob("*.xlsx"))
            if excel_files:
                return data_dir
    
    return None


def scan_data_directory(data_dir):
    """Scan data directory and report what files are found."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return None, []
    
    # Look for Excel files
    excel_files = list(data_path.glob("*.xlsx"))
    excel_files.extend(data_path.glob("*.xls"))
    
    # Check for expected files
    expected_files = ["cohort.xlsx", "iop.xlsx", "diagnosis.xlsx", "enc.xlsx"]
    found_files = {}
    
    for file in excel_files:
        filename = file.name.lower()
        for expected in expected_files:
            if expected.replace(".xlsx", "") in filename:
                found_files[expected] = file
                break
    
    return data_path, found_files


def main():
    """Main function to run aggregation."""
    print("=" * 60)
    print("Patient Data Aggregation Script")
    print("=" * 60)
    print()
    
    # Find data directory
    data_dir = find_data_directory()
    
    if data_dir is None:
        print("❌ Error: Could not find data directory with Excel files.")
        print()
        print("Please ensure one of these directories exists with Excel files:")
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
        print(f"❌ Error: No Excel files found in {data_dir}")
        return 1
    
    print("Found Excel files:")
    for expected_file in ["cohort.xlsx", "iop.xlsx", "diagnosis.xlsx", "enc.xlsx"]:
        if expected_file in found_files:
            print(f"  ✓ {expected_file}")
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
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: Required file not found: {e}")
        print("Please ensure all required Excel files are in the data directory.")
        return 1
    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

