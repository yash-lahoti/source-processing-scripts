#!/usr/bin/env python3
"""
Test script to verify large dataset (10K patients) processes correctly through the full pipeline.
Includes performance monitoring and detailed logging.
"""
import sys
import time
from pathlib import Path
import pandas as pd
import os

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from patient_aggregator.config_loader import load_config, get_feature_config, get_stratified_eda_config
from patient_aggregator.features import engineer_features
from patient_aggregator.stratified_eda import create_stratified_plots

def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    else:
        return 0.0  # Return 0 if psutil not available

def format_time(seconds):
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"

def main():
    """Run large dataset through full pipeline."""
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    print("=" * 60)
    print("Testing Large Dataset Processing (10K Patients)")
    print("=" * 60)
    print()
    
    # Load test dataset
    test_file = project_root / "test_data" / "aggregated_10k_patients.csv"
    
    if not test_file.exists():
        print(f"❌ Error: Dataset not found at {test_file}")
        print("   Please run: python scripts/create_large_synthetic_data.py")
        return 1
    
    print(f"✓ Loading dataset: {test_file}")
    load_start = time.time()
    df = pd.read_csv(test_file)
    load_time = time.time() - load_start
    load_memory = get_memory_usage()
    
    print(f"  - Patients: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Load time: {format_time(load_time)}")
    print(f"  - Memory after load: {load_memory:.1f} MB (+{load_memory - initial_memory:.1f} MB)")
    print()
    
    # Check data format
    print("Checking data format...")
    sample_col = 'bp_diastolic'
    if sample_col in df.columns:
        sample_value = df[sample_col].iloc[0]
        if isinstance(sample_value, str):
            try:
                import json
                parsed = json.loads(sample_value)
                print(f"  ✓ JSON parsing works: {parsed[:3]}... (sample)")
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
        feat_start = time.time()
        feat_memory_before = get_memory_usage()
        
        df_engineered, _ = engineer_features(df.copy(), feature_config)
        
        feat_time = time.time() - feat_start
        feat_memory_after = get_memory_usage()
        
        print()
        print("✓ Feature engineering completed!")
        print(f"  - Original columns: {len(df.columns)}")
        print(f"  - Final columns: {len(df_engineered.columns)}")
        print(f"  - New columns: {len(df_engineered.columns) - len(df.columns)}")
        print(f"  - Processing time: {format_time(feat_time)}")
        print(f"  - Memory: {feat_memory_after:.1f} MB (+{feat_memory_after - feat_memory_before:.1f} MB)")
        
        # Check for expected features
        expected_features = ['pulse_mean', 'od_iop_mean', 'os_iop_mean', 
                           'bp_systolic_mean', 'bp_diastolic_mean', 
                           'map', 'mopp_od', 'mopp_os']
        
        print()
        print("Checking computed features...")
        feature_stats = {}
        for feat in expected_features:
            if feat in df_engineered.columns:
                non_null = df_engineered[feat].notna().sum()
                pct = 100*non_null/len(df_engineered)
                feature_stats[feat] = {'available': True, 'non_null': non_null, 'pct': pct}
                status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "❌"
                print(f"  {status} {feat}: {non_null:,}/{len(df_engineered):,} ({pct:.1f}%)")
            else:
                feature_stats[feat] = {'available': False}
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
        output_dir = project_root / "test_data" / "large_dataset_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        
        print(f"✓ Generating plots to: {plots_dir}")
        plot_start = time.time()
        plot_memory_before = get_memory_usage()
        
        create_stratified_plots(df_engineered, stratified_config, plots_dir)
        
        plot_time = time.time() - plot_start
        plot_memory_after = get_memory_usage()
        
        print()
        print("✓ Plot generation completed!")
        print(f"  - Processing time: {format_time(plot_time)}")
        print(f"  - Memory: {plot_memory_after:.1f} MB (+{plot_memory_after - plot_memory_before:.1f} MB)")
        
        # Check if plots were created
        plot_files = list(plots_dir.rglob("*.png"))
        if plot_files:
            print(f"  - Generated {len(plot_files)} plot files")
            for plot_file in sorted(plot_files)[:5]:
                size_kb = plot_file.stat().st_size / 1024
                print(f"    ✓ {plot_file.name} ({size_kb:.1f} KB)")
            if len(plot_files) > 5:
                print(f"    ... and {len(plot_files) - 5} more")
        else:
            print("  ⚠ No plot files found")
        
        print()
        
    except Exception as e:
        print(f"❌ Error during plot generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print()
    print("✓ Dataset loaded successfully")
    print("✓ Feature engineering completed")
    print("✓ Plots generated")
    print()
    print("Performance Metrics:")
    print(f"  - Total time: {format_time(total_time)}")
    print(f"  - Data load: {format_time(load_time)}")
    print(f"  - Feature engineering: {format_time(feat_time)}")
    print(f"  - Plot generation: {format_time(plot_time)}")
    print(f"  - Peak memory: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)")
    print()
    print("Data retention check:")
    print(f"  - Original patients: {len(df):,}")
    print(f"  - Final patients: {len(df_engineered):,}")
    print(f"  - Patient retention: {100*len(df_engineered)/len(df):.1f}%")
    print()
    
    # Check for data loss in key features
    print("Feature data availability:")
    key_features = ['pulse_mean', 'od_iop_mean', 'os_iop_mean', 
                   'bp_systolic_mean', 'bp_diastolic_mean', 'map']
    all_good = True
    for feat in key_features:
        if feat in df_engineered.columns:
            available = df_engineered[feat].notna().sum()
            pct = 100 * available / len(df_engineered)
            status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "❌"
            if pct < 80:
                all_good = False
            print(f"  {status} {feat}: {available:,}/{len(df_engineered):,} ({pct:.1f}%)")
        else:
            all_good = False
            print(f"  ❌ {feat}: Missing!")
    
    print()
    print("=" * 60)
    if all_good:
        print("✓ All tests completed successfully!")
    else:
        print("⚠ Tests completed with warnings - check feature availability above")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

