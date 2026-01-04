"""Data cleaning module for filtering physiologically implausible values."""
import pandas as pd
import numpy as np
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
    
    for feature, limits in tqdm(thresholds.items(), desc="Applying thresholds", leave=False, unit="feature"):
        if feature not in df_cleaned.columns:
            continue
        
        min_val = limits.get('min', -np.inf)
        max_val = limits.get('max', np.inf)
        
        # Convert column to numeric if needed
        # Check if column is already numeric
        if not pd.api.types.is_numeric_dtype(df_cleaned[feature]):
            # Try to convert to numeric
            original_dtype = df_cleaned[feature].dtype
            df_cleaned[feature] = pd.to_numeric(df_cleaned[feature], errors='coerce')
            
            # Check if conversion was successful (at least some values converted)
            if df_cleaned[feature].isna().all():
                tqdm.write(f"  ⚠ {feature}: Cannot convert to numeric (all values invalid), skipping threshold check")
                continue
            elif df_cleaned[feature].isna().any():
                n_invalid = df_cleaned[feature].isna().sum()
                tqdm.write(f"  ⚠ {feature}: Converted to numeric, but {n_invalid} values could not be converted (set to NaN)")
        
        # Count values outside thresholds (only for numeric columns)
        n_outside = ((df_cleaned[feature] < min_val) | (df_cleaned[feature] > max_val)).sum()
        
        if n_outside > 0:
            stats[feature] = {
                'n_outside': int(n_outside),
                'n_total': len(df_cleaned),
                'min': min_val,
                'max': max_val
            }
            
            if action == "remove":
                # Remove rows with values outside thresholds
                df_cleaned = df_cleaned[(df_cleaned[feature] >= min_val) & (df_cleaned[feature] <= max_val)]
                tqdm.write(f"  ⚠ {feature}: Removed {n_outside} values outside [{min_val}, {max_val}]")
            elif action == "cap":
                # Clip values to thresholds
                df_cleaned[feature] = df_cleaned[feature].clip(lower=min_val, upper=max_val)
                tqdm.write(f"  ⚠ {feature}: Capped {n_outside} values to [{min_val}, {max_val}]")
            else:  # warn
                tqdm.write(f"  ⚠ {feature}: {n_outside}/{len(df_cleaned)} values outside [{min_val}, {max_val}]")
    
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
    thresholds = cleaning_config.get('thresholds', {})
    
    df_cleaned = df.copy()
    all_stats = {}
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Data Cleaning")
    tqdm.write("="*60 + "\n")
    
    # Apply threshold-based cleaning
    if method in ['threshold', 'both']:
        df_cleaned, threshold_stats = apply_cleaning_thresholds(df_cleaned, thresholds, action)
        all_stats['thresholds'] = threshold_stats
    
    # Apply IQR-based outlier detection
    if method in ['iqr', 'both']:
        iqr_stats = {}
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            if col in ['patient_uid']:  # Skip ID columns
                continue
            
            data = df_cleaned[col].dropna().values
            if len(data) == 0:
                continue
            
            outliers = detect_outliers_iqr(data, iqr_multiplier)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                iqr_stats[col] = {
                    'n_outliers': int(n_outliers),
                    'n_total': len(data),
                    'percentage': 100 * n_outliers / len(data)
                }
                
                if action == "remove":
                    outlier_indices = df_cleaned[col].dropna().index[outliers]
                    df_cleaned.loc[outlier_indices, col] = np.nan
                    tqdm.write(f"  ⚠ {col}: Removed {n_outliers} IQR outliers (multiplier={iqr_multiplier})")
                elif action == "warn":
                    tqdm.write(f"  ⚠ {col}: {n_outliers}/{len(data)} IQR outliers detected (multiplier={iqr_multiplier})")
        
        all_stats['iqr'] = iqr_stats
    
    tqdm.write("\n✓ Data cleaning complete!")
    
    return df_cleaned, all_stats

