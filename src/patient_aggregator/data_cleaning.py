"""Data cleaning module for filtering physiologically implausible values."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm


def apply_cleaning_thresholds(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]], 
                             action: str = "warn") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply min/max thresholds to clean data.
    
    Args:
        df: DataFrame with features
        thresholds: Dictionary mapping feature names to {min, max} thresholds
        action: "warn", "remove", or "cap"
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dictionary)
    """
    df_cleaned = df.copy()
    stats = {}
    initial_row_count = len(df_cleaned)
    
    tqdm.write(f"  Starting threshold cleaning on {initial_row_count} rows with {len(thresholds)} threshold rules")
    
    for feature, limits in tqdm(thresholds.items(), desc="Applying thresholds", leave=False, unit="feature"):
        if feature not in df_cleaned.columns:
            tqdm.write(f"  ⚠ {feature}: Column not found in DataFrame, skipping")
            continue
        
        min_val = limits.get('min', -np.inf)
        max_val = limits.get('max', np.inf)
        
        # Get initial statistics
        initial_non_null = df_cleaned[feature].notna().sum()
        initial_null = df_cleaned[feature].isna().sum()
        initial_total = len(df_cleaned)
        
        tqdm.write(f"\n  Processing {feature}:")
        tqdm.write(f"    Initial state: {initial_non_null} non-null, {initial_null} null, {initial_total} total rows")
        
        # Check if this is a JSON array column (string columns that look like JSON arrays)
        # These should be skipped - they will be processed after feature engineering
        if not pd.api.types.is_numeric_dtype(df_cleaned[feature]):
            # Check if values look like JSON arrays
            sample_values = df_cleaned[feature].dropna().head(3)
            if len(sample_values) > 0:
                first_val = str(sample_values.iloc[0])
                if first_val.strip().startswith('[') and first_val.strip().endswith(']'):
                    tqdm.write(f"    ⚠ {feature}: Appears to be JSON array column (e.g., '{first_val[:50]}...')")
                    tqdm.write(f"      Skipping - threshold cleaning should be applied after feature engineering (to _mean columns)")
                    continue
        
        # Convert column to numeric if needed
        # Check if column is already numeric
        if not pd.api.types.is_numeric_dtype(df_cleaned[feature]):
            original_dtype = df_cleaned[feature].dtype
            tqdm.write(f"    Converting from {original_dtype} to numeric...")
            
            # Try to convert to numeric
            df_cleaned[feature] = pd.to_numeric(df_cleaned[feature], errors='coerce')
            
            # Check conversion results
            after_conversion_non_null = df_cleaned[feature].notna().sum()
            after_conversion_null = df_cleaned[feature].isna().sum()
            conversion_lost = initial_non_null - after_conversion_non_null
            
            if df_cleaned[feature].isna().all():
                tqdm.write(f"    ❌ Conversion failed: All {initial_non_null} values became NaN, skipping threshold check")
                continue
            elif conversion_lost > 0:
                tqdm.write(f"    ⚠ Conversion: {conversion_lost} values could not be converted to numeric (now NaN)")
                tqdm.write(f"      After conversion: {after_conversion_non_null} numeric, {after_conversion_null} null")
        
        # Get pre-threshold statistics
        pre_threshold_data = df_cleaned[feature].dropna()
        if len(pre_threshold_data) > 0:
            pre_min = pre_threshold_data.min()
            pre_max = pre_threshold_data.max()
            pre_mean = pre_threshold_data.mean()
            pre_median = pre_threshold_data.median()
            tqdm.write(f"    Pre-threshold stats: min={pre_min:.2f}, max={pre_max:.2f}, mean={pre_mean:.2f}, median={pre_median:.2f}")
        
        # Count values outside thresholds (only for numeric columns)
        mask_below = df_cleaned[feature] < min_val
        mask_above = df_cleaned[feature] > max_val
        mask_outside = mask_below | mask_above
        n_below = mask_below.sum()
        n_above = mask_above.sum()
        n_outside = mask_outside.sum()
        n_within = (~mask_outside & df_cleaned[feature].notna()).sum()
        
        tqdm.write(f"    Threshold: [{min_val}, {max_val}]")
        tqdm.write(f"    Values below threshold: {n_below}")
        tqdm.write(f"    Values above threshold: {n_above}")
        tqdm.write(f"    Values outside threshold: {n_outside} ({100*n_outside/initial_non_null:.1f}% of non-null)")
        tqdm.write(f"    Values within threshold: {n_within} ({100*n_within/initial_non_null:.1f}% of non-null)")
        
        if n_outside > 0:
            # Show example values that are outside
            outside_values = df_cleaned.loc[mask_outside, feature]
            if len(outside_values) > 0:
                example_below = outside_values[outside_values < min_val]
                example_above = outside_values[outside_values > max_val]
                if len(example_below) > 0:
                    tqdm.write(f"    Example values below threshold: {example_below.head(3).tolist()}")
                if len(example_above) > 0:
                    tqdm.write(f"    Example values above threshold: {example_above.head(3).tolist()}")
            
            stats[feature] = {
                'n_outside': int(n_outside),
                'n_below': int(n_below),
                'n_above': int(n_above),
                'n_total': initial_total,
                'n_non_null_before': int(initial_non_null),
                'min': min_val,
                'max': max_val
            }
            
            if action == "remove":
                # Remove rows with values outside thresholds
                rows_before = len(df_cleaned)
                df_cleaned = df_cleaned[(df_cleaned[feature] >= min_val) & (df_cleaned[feature] <= max_val)]
                rows_after = len(df_cleaned)
                rows_removed = rows_before - rows_after
                tqdm.write(f"    ❌ REMOVED {rows_removed} rows ({100*rows_removed/rows_before:.1f}% of total)")
                tqdm.write(f"      Rows remaining: {rows_after} (from {rows_before})")
            elif action == "cap":
                # Clip values to thresholds
                n_capped_below = n_below
                n_capped_above = n_above
                df_cleaned[feature] = df_cleaned[feature].clip(lower=min_val, upper=max_val)
                tqdm.write(f"    ⚠ CAPPED {n_outside} values:")
                tqdm.write(f"      {n_capped_below} values capped to min={min_val}")
                tqdm.write(f"      {n_capped_above} values capped to max={max_val}")
                
                # Show post-capping stats
                post_cap_data = df_cleaned[feature].dropna()
                if len(post_cap_data) > 0:
                    tqdm.write(f"    Post-capping stats: min={post_cap_data.min():.2f}, max={post_cap_data.max():.2f}")
            else:  # warn
                tqdm.write(f"    ⚠ WARNING: {n_outside}/{initial_non_null} values outside threshold (no action taken)")
        else:
            tqdm.write(f"    ✓ All {initial_non_null} non-null values within threshold")
    
    final_row_count = len(df_cleaned)
    rows_lost = initial_row_count - final_row_count
    if rows_lost > 0:
        tqdm.write(f"\n  Summary: {rows_lost} rows removed ({100*rows_lost/initial_row_count:.1f}% of initial {initial_row_count} rows)")
        tqdm.write(f"  Final row count: {final_row_count}")
    else:
        tqdm.write(f"\n  Summary: No rows removed, {final_row_count} rows remain")
    
    return df_cleaned, stats


def detect_outliers_iqr(data: np.ndarray, multiplier: float = 3.0) -> np.ndarray:
    """
    Detect outliers using IQR method.
    
    Args:
        data: Array of values
        multiplier: IQR multiplier (1.5 for mild, 3.0 for strict)
        
    Returns:
        Boolean array indicating outliers
    """
    if len(data) == 0:
        return np.array([])
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        return np.zeros(len(data), dtype=bool)
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (data < lower_bound) | (data > upper_bound)


def clean_feature_data(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to clean feature data based on configuration.
    
    Args:
        df: DataFrame with features
        config: Feature configuration dictionary
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    cleaning_config = config.get('data_cleaning', {})
    
    if not cleaning_config.get('enabled', False):
        return df, {}
    
    method = cleaning_config.get('method', 'both')
    action = cleaning_config.get('action', 'warn')
    iqr_multiplier = cleaning_config.get('iqr_multiplier', 3.0)
    
    df_cleaned = df.copy()
    all_stats = {}
    
    # Initial statistics
    initial_rows = len(df_cleaned)
    initial_cols = len(df_cleaned.columns)
    initial_total_values = df_cleaned.size
    initial_non_null = df_cleaned.notna().sum().sum()
    initial_null = df_cleaned.isna().sum().sum()
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Data Cleaning")
    tqdm.write("="*60 + "\n")
    tqdm.write(f"  Initial dataset: {initial_rows} rows × {initial_cols} columns")
    tqdm.write(f"  Initial values: {initial_non_null} non-null, {initial_null} null ({100*initial_null/initial_total_values:.1f}% null)")
    tqdm.write(f"  Cleaning method: {method} (threshold-based cleaning disabled)")
    tqdm.write(f"  Action: {action}")
    
    # Threshold-based cleaning removed - it fails on JSON array columns before feature engineering
    # Only IQR-based outlier detection is used
    if method in ['threshold', 'both']:
        tqdm.write(f"\n  ⚠ Threshold-based cleaning skipped (disabled to avoid errors on JSON array columns)")
        tqdm.write(f"     Threshold cleaning should be applied after feature engineering if needed")
    
    # Apply IQR-based outlier detection
    if method in ['iqr', 'both']:
        tqdm.write(f"\n  IQR Outlier Detection (multiplier={iqr_multiplier}):")
        numeric_cols = [col for col in df_cleaned.select_dtypes(include=[np.number]).columns 
                       if col not in ['patient_uid']]
        tqdm.write(f"  Processing {len(numeric_cols)} numeric columns for IQR outliers")
        
        iqr_stats = {}
        total_outliers_removed = 0
        
        for col in numeric_cols:
            initial_non_null = df_cleaned[col].notna().sum()
            if initial_non_null == 0:
                continue
            
            data = df_cleaned[col].dropna().values
            if len(data) < 4:  # Need at least 4 values for meaningful IQR
                tqdm.write(f"    {col}: Skipped (only {len(data)} values, need 4+ for IQR)")
                continue
            
            # Calculate IQR statistics
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            median = np.median(data)
            mean = np.mean(data)
            
            if iqr == 0:
                tqdm.write(f"    {col}: Skipped (IQR=0, all values identical)")
                continue
            
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            outliers = detect_outliers_iqr(data, iqr_multiplier)
            n_outliers = outliers.sum()
            outlier_percentage = 100 * n_outliers / len(data)
            
            tqdm.write(f"\n    {col}:")
            tqdm.write(f"      Data stats: n={len(data)}, mean={mean:.2f}, median={median:.2f}")
            tqdm.write(f"      IQR: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
            tqdm.write(f"      Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] (multiplier={iqr_multiplier})")
            
            if n_outliers > 0:
                outlier_values = data[outliers]
                outliers_below = (outlier_values < lower_bound).sum()
                outliers_above = (outlier_values > upper_bound).sum()
                
                tqdm.write(f"      Outliers: {n_outliers} ({outlier_percentage:.1f}%)")
                tqdm.write(f"        Below lower bound: {outliers_below}")
                tqdm.write(f"        Above upper bound: {outliers_above}")
                
                if len(outlier_values) > 0:
                    tqdm.write(f"      Example outlier values: {outlier_values[:5].tolist()}")
                
                iqr_stats[col] = {
                    'n_outliers': int(n_outliers),
                    'n_total': len(data),
                    'percentage': outlier_percentage,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr)
                }
                
                if action == "remove":
                    outlier_indices = df_cleaned[col].dropna().index[outliers]
                    df_cleaned.loc[outlier_indices, col] = np.nan
                    total_outliers_removed += n_outliers
                    after_non_null = df_cleaned[col].notna().sum()
                    tqdm.write(f"      ❌ REMOVED {n_outliers} outliers ({100*n_outliers/initial_non_null:.1f}% of non-null)")
                    tqdm.write(f"      Remaining non-null: {after_non_null} (from {initial_non_null})")
                elif action == "warn":
                    tqdm.write(f"      ⚠ WARNING: {n_outliers} outliers detected (no action taken)")
            else:
                tqdm.write(f"      ✓ No outliers detected")
        
        if total_outliers_removed > 0:
            tqdm.write(f"\n  IQR Summary: {total_outliers_removed} total outlier values removed")
        
        all_stats['iqr'] = iqr_stats
    
    # Final statistics
    final_rows = len(df_cleaned)
    final_cols = len(df_cleaned.columns)
    final_total_values = df_cleaned.size
    final_non_null = df_cleaned.notna().sum().sum()
    final_null = df_cleaned.isna().sum().sum()
    
    rows_lost = initial_rows - final_rows
    values_lost = initial_non_null - final_non_null
    null_increase = final_null - initial_null
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Data Cleaning Summary")
    tqdm.write("="*60)
    tqdm.write(f"  Rows: {initial_rows} → {final_rows} ({rows_lost} removed, {100*rows_lost/initial_rows:.1f}%)")
    tqdm.write(f"  Columns: {initial_cols} → {final_cols}")
    tqdm.write(f"  Non-null values: {initial_non_null} → {final_non_null} ({values_lost} removed, {100*values_lost/initial_non_null:.1f}%)")
    tqdm.write(f"  Null values: {initial_null} → {final_null} ({null_increase:+d} change)")
    tqdm.write(f"  Overall data retention: {100*final_non_null/initial_non_null:.1f}% of non-null values")
    
    if values_lost > initial_non_null * 0.1:  # More than 10% lost
        tqdm.write(f"\n  ⚠ WARNING: Significant data loss detected ({100*values_lost/initial_non_null:.1f}% of values removed)")
        tqdm.write(f"  Consider reviewing cleaning thresholds and IQR multiplier settings")
    
    tqdm.write("\n✓ Data cleaning complete!\n")
    
    # Add overall summary to stats for reporting
    all_stats['overall'] = {
        'initial_rows': int(initial_rows),
        'final_rows': int(final_rows),
        'initial_non_null': int(initial_non_null),
        'final_non_null': int(final_non_null),
        'rows_lost': int(rows_lost),
        'values_lost': int(values_lost),
        'null_increase': int(null_increase),
        'retention_percentage': float(100 * final_non_null / initial_non_null) if initial_non_null > 0 else 0.0
    }
    
    return df_cleaned, all_stats


def save_cleaning_report(cleaning_stats: Dict[str, Any], output_dir: Path):
    """
    Save cleaning impact report as CSV and text files.
    
    Args:
        cleaning_stats: Statistics dictionary from clean_feature_data()
        output_dir: Output directory to save reports
    """
    if not cleaning_stats:
        tqdm.write("  ⚠ No cleaning statistics available, skipping report generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract IQR stats
    iqr_stats = cleaning_stats.get('iqr', {})
    overall_stats = cleaning_stats.get('overall', {})
    
    # Create CSV report
    csv_data = []
    for feature, stats in iqr_stats.items():
        csv_data.append({
            'Feature': feature,
            'Initial_Rows': overall_stats.get('initial_rows', 0),
            'Final_Rows': overall_stats.get('final_rows', 0),
            'Initial_Non_Null': stats.get('n_total', 0),
            'Final_Non_Null': stats.get('n_total', 0) - stats.get('n_outliers', 0),
            'Outliers_Detected': stats.get('n_outliers', 0),
            'Outlier_Percentage': stats.get('percentage', 0.0),
            'Lower_Bound': stats.get('lower_bound', 0.0),
            'Upper_Bound': stats.get('upper_bound', 0.0),
            'Q1': stats.get('q1', 0.0),
            'Q3': stats.get('q3', 0.0),
            'IQR': stats.get('iqr', 0.0)
        })
    
    if csv_data:
        csv_path = output_dir / "cleaning_impact_report.csv"
        import pandas as pd
        df_report = pd.DataFrame(csv_data)
        df_report.to_csv(csv_path, index=False, float_format='%.3f')
        tqdm.write(f"  ✓ Saved: {csv_path.name}")
    
    # Create text report
    txt_path = output_dir / "cleaning_impact_report.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLEANING IMPACT REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-"*70 + "\n")
        initial_rows = overall_stats.get('initial_rows', 0)
        final_rows = overall_stats.get('final_rows', 0)
        rows_lost = overall_stats.get('rows_lost', 0)
        initial_non_null = overall_stats.get('initial_non_null', 0)
        final_non_null = overall_stats.get('final_non_null', 0)
        values_lost = overall_stats.get('values_lost', 0)
        
        f.write(f"Initial rows: {initial_rows}\n")
        f.write(f"Final rows: {final_rows}\n")
        if initial_rows > 0:
            f.write(f"Rows removed: {rows_lost} ({100*rows_lost/initial_rows:.1f}%)\n")
        else:
            f.write(f"Rows removed: {rows_lost}\n")
        f.write("\n")
        
        f.write(f"Initial non-null values: {initial_non_null}\n")
        f.write(f"Final non-null values: {final_non_null}\n")
        if initial_non_null > 0:
            f.write(f"Values removed: {values_lost} ({100*values_lost/initial_non_null:.1f}%)\n")
        else:
            f.write(f"Values removed: {values_lost}\n")
        f.write(f"Data retention: {overall_stats.get('retention_percentage', 0.0):.1f}%\n\n")
        
        # Per-feature breakdown
        if iqr_stats:
            f.write("="*70 + "\n")
            f.write("PER-FEATURE BREAKDOWN (IQR Outlier Detection)\n")
            f.write("="*70 + "\n\n")
            
            for feature, stats in sorted(iqr_stats.items()):
                f.write(f"Feature: {feature}\n")
                f.write("-"*70 + "\n")
                f.write(f"  Total values: {stats.get('n_total', 0)}\n")
                f.write(f"  Outliers detected: {stats.get('n_outliers', 0)} ({stats.get('percentage', 0.0):.1f}%)\n")
                f.write(f"  IQR statistics:\n")
                f.write(f"    Q1: {stats.get('q1', 0.0):.2f}\n")
                f.write(f"    Q3: {stats.get('q3', 0.0):.2f}\n")
                f.write(f"    IQR: {stats.get('iqr', 0.0):.2f}\n")
                f.write(f"  Outlier bounds: [{stats.get('lower_bound', 0.0):.2f}, {stats.get('upper_bound', 0.0):.2f}]\n")
                f.write("\n")
        else:
            f.write("="*70 + "\n")
            f.write("PER-FEATURE BREAKDOWN\n")
            f.write("="*70 + "\n\n")
            f.write("No IQR outlier detection was performed or no outliers were detected.\n\n")
        
        f.write("="*70 + "\n")
        f.write("NOTE: Threshold-based cleaning has been disabled to avoid errors on\n")
        f.write("JSON array columns before feature engineering. Only IQR-based outlier\n")
        f.write("detection is used.\n")
        f.write("="*70 + "\n")
    
    tqdm.write(f"  ✓ Saved: {txt_path.name}")

