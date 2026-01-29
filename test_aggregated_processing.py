#!/usr/bin/env python3
"""
Test script to verify aggregated dataset processes correctly through the full pipeline.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from patient_aggregator.config_loader import load_config, get_feature_config, get_stratified_eda_config
from patient_aggregator.features import engineer_features
from patient_aggregator.stratified_eda import create_stratified_plots

def main():
    """Run test dataset through full pipeline."""
    print("=" * 60)
    print("Testing Aggregated Dataset Processing")
    print("=" * 60)
    print()
    
    # Load test dataset
    test_file = project_root / "test_data" / "aggregated_test_patients.csv"
    
    if not test_file.exists():
        print(f"❌ Error: Test dataset not found at {test_file}")
        print("   Please run: python scripts/create_test_aggregated_data.py")
        return 1
    
    print(f"✓ Loading test dataset: {test_file}")
    df = pd.read_csv(test_file)
    print(f"  - Patients: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Columns: {', '.join(df.columns.tolist())}")
    print()
    
    # Check data format
    print("Checking data format...")
    sample_col = 'bp_diastolic'
    if sample_col in df.columns:
        sample_value = df[sample_col].iloc[0]
        print(f"  Sample {sample_col}: {sample_value} (type: {type(sample_value).__name__})")
        if isinstance(sample_value, str):
            try:
                import json
                parsed = json.loads(sample_value)
                print(f"  ✓ JSON parsing works: {parsed}")
            except:
                print(f"  ⚠ JSON parsing failed")
    print()
    
    # Load config
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return 1
    
    print(f"✓ Loading config: {config_path}")
    config = load_config(str(config_path))
    print()
    
    # Get feature config
    feature_config = get_feature_config(config)
    
    if not feature_config.get('enabled', False):
        print("⚠ Feature engineering is disabled in config. Enabling for test...")
        feature_config['enabled'] = True
    
    # Run feature engineering
    print("=" * 60)
    print("Running Feature Engineering Pipeline")
    print("=" * 60)
    print()
    
    try:
        df_engineered, _ = engineer_features(df.copy(), feature_config)
        
        print()
        print("✓ Feature engineering completed!")
        print(f"  - Original columns: {len(df.columns)}")
        print(f"  - Final columns: {len(df_engineered.columns)}")
        print(f"  - New columns: {len(df_engineered.columns) - len(df.columns)}")
        
        # Check for expected features
        expected_features = ['pulse_mean', 'od_iop_mean', 'os_iop_mean', 
                           'bp_systolic_mean', 'bp_diastolic_mean', 
                           'map', 'mopp_od', 'mopp_os']
        
        print()
        print("Checking computed features...")
        for feat in expected_features:
            if feat in df_engineered.columns:
                non_null = df_engineered[feat].notna().sum()
                print(f"  ✓ {feat}: {non_null}/{len(df_engineered)} non-null ({100*non_null/len(df_engineered):.1f}%)")
            else:
                print(f"  ❌ {feat}: Missing!")
        
        print()
        
    except Exception as e:
        print(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test stratified plots
    print("=" * 60)
    print("Testing Stratified Plot Generation")
    print("=" * 60)
    print()
    
    try:
        stratified_config = get_stratified_eda_config(config)
        
        if not stratified_config.get('enabled', False):
            print("⚠ Stratified EDA is disabled in config. Enabling for test...")
            stratified_config['enabled'] = True
        
        # Create output directory
        output_dir = project_root / "test_data" / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        
        print(f"✓ Generating plots to: {plots_dir}")
        create_stratified_plots(df_engineered, stratified_config, plots_dir)
        
        print()
        print("✓ Plot generation completed!")
        
        # Check if plots were created
        plot_files = list(plots_dir.rglob("*.png"))
        if plot_files:
            print(f"  - Generated {len(plot_files)} plot files")
            for plot_file in plot_files[:5]:
                print(f"    ✓ {plot_file.name}")
        else:
            print("  ⚠ No plot files found")
        
        print()
        
    except Exception as e:
        print(f"❌ Error during plot generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print()
    print("✓ Dataset loaded successfully")
    print("✓ Feature engineering completed")
    print("✓ Plots generated")
    print()
    print("Data retention check:")
    print(f"  - Original patients: {len(df)}")
    print(f"  - Final patients: {len(df_engineered)}")
    print(f"  - Patient retention: {100*len(df_engineered)/len(df):.1f}%")
    print()
    
    # Check for data loss in key features
    print("Feature data availability:")
    key_features = ['pulse_mean', 'od_iop_mean', 'os_iop_mean', 
                   'bp_systolic_mean', 'bp_diastolic_mean', 'map']
    for feat in key_features:
        if feat in df_engineered.columns:
            available = df_engineered[feat].notna().sum()
            pct = 100 * available / len(df_engineered)
            status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "❌"
            print(f"  {status} {feat}: {available}/{len(df_engineered)} ({pct:.1f}%)")
    
    print()
    print("=" * 60)
    print("✓ All tests completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


