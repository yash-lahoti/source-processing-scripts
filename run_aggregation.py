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
    """Main function to run aggregation with detailed logging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate patient data and generate visualizations")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating visualizations")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Patient Data Aggregation Script")
    print("=" * 60)
    print()
    
    # ========================================================================
    # STEP 1: Find and scan data directory
    # ========================================================================
    print("=" * 60)
    print("STEP 1: Data Directory Discovery")
    print("=" * 60)
    
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
    
    # Scan for files
    data_path, found_files = scan_data_directory(data_dir)
    
    if not found_files:
        print(f"❌ Error: No data files (.xlsx, .xls, or .csv) found in {data_dir}")
        print(f"   Please ensure files like 'cohort', 'iop', 'diagnosis', or 'enc' exist in this directory")
        return 1
    
    print("\nFound data files:")
    for expected_file in ["cohort.xlsx", "iop.xlsx", "diagnosis.xlsx", "enc.xlsx"]:
        if expected_file in found_files:
            actual_file = found_files[expected_file]
            print(f"  ✓ {expected_file} (found as {actual_file.name})")
        else:
            print(f"  ⚠ {expected_file} (not found - will be skipped)")
    print()
    
    # ========================================================================
    # STEP 2: Load configuration and setup output directory
    # ========================================================================
    print("=" * 60)
    print("STEP 2: Configuration and Output Setup")
    print("=" * 60)
    
    from patient_aggregator.config_loader import load_config, get_output_directory
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.yaml"
    
    config = load_config(str(config_path)) if config_path.exists() else {}
    
    if config_path.exists():
        print(f"✓ Loaded configuration from: {config_path}")
    else:
        print("⚠ No config.yaml found, using package defaults")
    
    # Create output directory
    output_dir = get_output_directory(config, Path(__file__).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Set output file in output directory
    output_filename = config.get('output', {}).get('aggregated_file', 'aggregated_patients.csv')
    output_file = output_dir / output_filename
    print(f"✓ Output file: {output_file}")
    print()
    
    # ========================================================================
    # STEP 3: Run aggregation
    # ========================================================================
    print("=" * 60)
    print("STEP 3: Data Aggregation")
    print("=" * 60)
    
    if output_file.exists():
        print("✓ Aggregated file already exists. Skipping aggregation.")
        print("  (Delete the file or use --force to regenerate)")
        print()
    else:
        print("Starting aggregation...")
        print("-" * 60)
    
    try:
        # Run aggregation (will skip if file exists)
        aggregate_patients(
            input_dir=str(data_path),
            output_file=str(output_file)
        )
        
        if output_file.exists():
            import pandas as pd
            df = pd.read_csv(output_file)
            print(f"\n✓ Aggregation complete!")
            print(f"  File: {output_file}")
            print(f"  Patients: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print()
        else:
            print("\n⚠ Warning: Aggregation completed but output file not found")
            print()
        
    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 4: Filtering and subsetting
    # ========================================================================
    print("=" * 60)
    print("STEP 4: Filtering and Subsetting")
    print("=" * 60)
    
    try:
        import pandas as pd
        from patient_aggregator.config_loader import get_filter_config
        from patient_aggregator.filter import create_subset, save_subset
        
        df = pd.read_csv(output_file)
        print(f"✓ Loaded aggregated data: {len(df)} patients, {len(df.columns)} columns")
        
        if not config_path.exists():
            print("⚠ No config file found, skipping filtering")
            subset_df = df
            subset_file = None
        else:
            config = load_config(str(config_path))
            filter_config = get_filter_config(config)
            
            if filter_config.get('enabled', False):
                print("✓ Filtering enabled in config")
                
                subset_filename = filter_config.get('subset_file', 'aggregated_patients_subset.csv')
                subset_file = output_dir / subset_filename
                
                # Only create subset if it doesn't exist or if aggregation was just run
                if not subset_file.exists() or (output_file.exists() and output_file.stat().st_mtime > subset_file.stat().st_mtime):
                    print("  Creating filtered subset...")
                    subset_df = create_subset(df, filter_config)
                    save_subset(subset_df, str(subset_file))
                    print(f"✓ Subset saved: {subset_file}")
                    print(f"  Patients: {len(df)} → {len(subset_df)} ({100*len(subset_df)/len(df):.1f}% retained)")
                else:
                    subset_df = pd.read_csv(subset_file)
                    print(f"✓ Subset file already exists: {subset_file.name}")
                    print(f"  Patients: {len(subset_df)}")
            else:
                print("⚠ Filtering disabled in config, using full dataset")
                subset_df = df
                subset_file = None
        
        print()
        
    except Exception as e:
        print(f"❌ Error during filtering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 5: Feature engineering and cleaning
    # ========================================================================
    print("=" * 60)
    print("STEP 5: Feature Engineering and Cleaning")
    print("=" * 60)
    
    cleaning_stats = None
    full_with_features_df = None  # All patients with features (for stratified EDA: all_patients)

    try:
        if not config_path.exists():
            print("⚠ No config file found, skipping feature engineering")
            final_cohort_df = subset_df
            full_with_features_df = subset_df  # Same as final when no config
            cleaning_stats = None
        else:
            config = load_config(str(config_path))
            from patient_aggregator.config_loader import get_feature_config
            from patient_aggregator.features import engineer_features

            feature_config = get_feature_config(config)

            if feature_config.get('enabled', False):
                print("✓ Feature engineering enabled in config")
                # Subgroup: patients who meet inclusion criteria (with features)
                print(f"  Subgroup (inclusion criteria met): {len(subset_df)} patients, {len(subset_df.columns)} columns")
                final_cohort_df, cleaning_stats = engineer_features(subset_df, feature_config)
                print(f"  Output: {len(final_cohort_df)} patients, {len(final_cohort_df.columns)} columns")

                # Full set: all patients with features (same analysis pipeline for "all patients")
                if len(df) != len(subset_df):
                    print(f"  All patients: {len(df)} patients -> engineering features...")
                    full_with_features_df, _ = engineer_features(df, feature_config)
                    print(f"  Output: {len(full_with_features_df)} patients, {len(full_with_features_df.columns)} columns")
                else:
                    full_with_features_df = final_cohort_df
                print()
            else:
                print("⚠ Feature engineering disabled in config")
                final_cohort_df = subset_df
                full_with_features_df = subset_df
                cleaning_stats = None

    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 6: Save cohorts and generate cleaning report
    # ========================================================================
    print("=" * 60)
    print("STEP 6: Save Cohorts and Cleaning Report")
    print("=" * 60)

    try:
        # Save subgroup: patients who meet inclusion criteria (measurements specified in config)
        inclusion_file = output_dir / "aggregated_patients_inclusion_criteria_met_with_features.csv"
        final_cohort_df.to_csv(inclusion_file, index=False)
        print(f"✓ Subgroup (inclusion criteria met) saved: {inclusion_file.name}")
        print(f"  Patients: {len(final_cohort_df)}")
        print(f"  Columns: {len(final_cohort_df.columns)}")

        # Save full set: all patients with features (same pipeline for stratified EDA)
        if full_with_features_df is not None:
            all_file = output_dir / "aggregated_patients_all_with_features.csv"
            full_with_features_df.to_csv(all_file, index=False)
            print(f"✓ All patients saved: {all_file.name}")
            print(f"  Patients: {len(full_with_features_df)}")
            print(f"  Columns: {len(full_with_features_df.columns)}")
        print()

        # Generate cleaning report if cleaning was performed
        if cleaning_stats:
            print("Generating cleaning impact report...")
            from patient_aggregator.data_cleaning import save_cleaning_report
            save_cleaning_report(cleaning_stats, output_dir)
            print()
        else:
            print("⚠ No cleaning statistics available (cleaning may be disabled)")
            print()

    except Exception as e:
        print(f"❌ Error saving cohorts: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 7: Generate stratified plots (all patients + inclusion-criteria met)
    # ========================================================================
    if not args.no_plots:
        print("=" * 60)
        print("STEP 7: Generate Stratified Summary Plots")
        print("=" * 60)

        try:
            if not config_path.exists():
                print("⚠ No config file found, skipping stratified plots")
            else:
                config = load_config(str(config_path))
                from patient_aggregator.config_loader import get_stratified_eda_config
                from patient_aggregator.stratified_eda import create_stratified_plots

                stratified_config = get_stratified_eda_config(config)

                if stratified_config.get('enabled', False):
                    print("✓ Stratified EDA enabled in config (prescribed severity used for both)")
                    plots_dir = output_dir / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    # All patients: same analysis on full set
                    all_df = full_with_features_df if full_with_features_df is not None else final_cohort_df
                    all_patients_dir = plots_dir / "all_patients"
                    all_patients_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  All patients: {len(all_df)} -> {all_patients_dir.relative_to(output_dir)}")
                    create_stratified_plots(all_df, stratified_config, all_patients_dir)
                    print()

                    # Inclusion criteria met: same analysis on subgroup
                    inclusion_dir = plots_dir / "inclusion_criteria_met"
                    inclusion_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  Inclusion criteria met: {len(final_cohort_df)} -> {inclusion_dir.relative_to(output_dir)}")
                    create_stratified_plots(final_cohort_df, stratified_config, inclusion_dir)
                    print()
                else:
                    print("⚠ Stratified EDA disabled in config, skipping plots")
                    print()
        except Exception as e:
            print(f"⚠ Warning: Could not generate visualizations: {e}")
            print("Aggregation completed successfully, but visualizations were skipped.")
            import traceback
            traceback.print_exc()
            print()
    else:
        print("=" * 60)
        print("STEP 7: Visualization Generation")
        print("=" * 60)
        print("⚠ Skipped (--no-plots flag set)")
        print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)
    print("Cohorts (prescribed severity used in stratified EDA):")
    print(f"  - All patients: {output_dir / 'aggregated_patients_all_with_features.csv'}")
    print(f"    N = {len(full_with_features_df) if full_with_features_df is not None else len(final_cohort_df)}")
    print(f"  - Inclusion criteria met: {output_dir / 'aggregated_patients_inclusion_criteria_met_with_features.csv'}")
    print(f"    N = {len(final_cohort_df)}")
    print()
    print("Output files:")
    print(f"  - {output_file.name} (raw aggregation)")
    if subset_file:
        print(f"  - {subset_file.name} (filtered subset)")
    print(f"  - aggregated_patients_all_with_features.csv (all patients, with features)")
    print(f"  - aggregated_patients_inclusion_criteria_met_with_features.csv (inclusion criteria met, with features)")
    if not args.no_plots:
        print(f"  - plots/all_patients/stratified_plots/ (figures and tables: all patients)")
        print(f"  - plots/inclusion_criteria_met/stratified_plots/ (figures and tables: inclusion criteria met)")
    print()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

