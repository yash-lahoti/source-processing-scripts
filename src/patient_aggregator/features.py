"""Feature engineering module for patient data."""
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm


def parse_json_array(value: str) -> List:
    """Parse a JSON array string into a Python list."""
    if pd.isna(value) or value == '' or value == '[]':
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        return []


def compute_patient_means(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compute mean values for each patient across specified columns.
    
    Args:
        df: DataFrame with aggregated patient data
        columns: List of column names to compute means for
        
    Returns:
        DataFrame with new mean columns added
    """
    df = df.copy()
    
    for col in tqdm(columns, desc="Computing means", leave=False, unit="col"):
        if col not in df.columns:
            tqdm.write(f"  ⚠ Warning: Column '{col}' not found, skipping")
            continue
        
        mean_col = f"{col}_mean"
        df[mean_col] = df[col].apply(lambda x: np.mean(parse_json_array(x)) if parse_json_array(x) else np.nan)
    
    return df


def compute_variability(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compute standard deviation (variability) for each patient across specified columns.
    Only computes if patient has 2+ measurements.
    
    Args:
        df: DataFrame with aggregated patient data
        columns: List of column names to compute variability for
        
    Returns:
        DataFrame with new std columns added
    """
    df = df.copy()
    
    for col in tqdm(columns, desc="Computing variability", leave=False, unit="col"):
        if col not in df.columns:
            tqdm.write(f"  ⚠ Warning: Column '{col}' not found, skipping")
            continue
        
        std_col = f"{col}_std"
        
        def compute_std(values_str):
            values = parse_json_array(values_str)
            if len(values) < 2:
                return np.nan  # Need at least 2 values for std
            return np.std(values)
        
        df[std_col] = df[col].apply(compute_std)
    
    return df


def compute_derived_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute derived physiological features based on formulas in config.
    
    Args:
        df: DataFrame with mean columns already computed
        config: Feature configuration dictionary
        
    Returns:
        DataFrame with derived features added
    """
    df = df.copy()
    
    derived_features = config.get('derived_features', {})
    
    if not derived_features:
        return df
    
    tqdm.write("Computing derived features...")
    
    # Compute MAP first (needed for MOPP)
    if 'map' in derived_features:
        map_config = derived_features['map']
        formula = map_config.get('formula', '')
        
        # Replace column names in formula with actual column references
        # Formula: "bp_diastolic_mean + (1/3) * (bp_systolic_mean - bp_diastolic_mean)"
        try:
            df['map'] = df['bp_diastolic_mean'] + (1/3) * (df['bp_systolic_mean'] - df['bp_diastolic_mean'])
            tqdm.write(f"  ✓ MAP computed")
        except Exception as e:
            tqdm.write(f"  ⚠ Error computing MAP: {e}")
    
    # Compute other derived features
    for feat_name, feat_config in derived_features.items():
        if feat_name == 'map':
            continue  # Already computed
        
        formula = feat_config.get('formula', '')
        description = feat_config.get('description', feat_name)
        
        try:
            # Parse formula and compute
            # Handle formulas like "(2/3) * map - od_iop_mean"
            if 'mopp_od' in feat_name:
                df['mopp_od'] = (2/3) * df['map'] - df['od_iop_mean']
            elif 'mopp_os' in feat_name:
                df['mopp_os'] = (2/3) * df['map'] - df['os_iop_mean']
            elif 'sopp_od' in feat_name:
                df['sopp_od'] = df['bp_systolic_mean'] - df['od_iop_mean']
            elif 'sopp_os' in feat_name:
                df['sopp_os'] = df['bp_systolic_mean'] - df['os_iop_mean']
            elif 'dopp_od' in feat_name:
                df['dopp_od'] = df['bp_diastolic_mean'] - df['od_iop_mean']
            elif 'dopp_os' in feat_name:
                df['dopp_os'] = df['bp_diastolic_mean'] - df['os_iop_mean']
            else:
                # Generic formula evaluation (if needed for future features)
                # Replace common patterns
                formula_eval = formula.replace('map', 'df["map"]')
                formula_eval = formula_eval.replace('bp_systolic_mean', 'df["bp_systolic_mean"]')
                formula_eval = formula_eval.replace('bp_diastolic_mean', 'df["bp_diastolic_mean"]')
                formula_eval = formula_eval.replace('od_iop_mean', 'df["od_iop_mean"]')
                formula_eval = formula_eval.replace('os_iop_mean', 'df["os_iop_mean"]')
                df[feat_name] = eval(formula_eval)
            
            tqdm.write(f"  ✓ {feat_name} ({description}) computed")
        except Exception as e:
            tqdm.write(f"  ⚠ Error computing {feat_name}: {e}")
    
    return df


def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main function to orchestrate feature engineering.
    
    Args:
        df: Input DataFrame with aggregated patient data
        config: Feature engineering configuration
        
    Returns:
        Enhanced DataFrame with all computed features
    """
    if not config.get('enabled', False):
        return df
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Feature Engineering Pipeline")
    tqdm.write("="*60 + "\n")
    
    # Compute means
    if config.get('compute_means', True):
        mean_columns = config.get('mean_columns', [])
        if mean_columns:
            df = compute_patient_means(df, mean_columns)
    
    # Compute variability
    if config.get('compute_variability', True):
        mean_columns = config.get('mean_columns', [])
        if mean_columns:
            df = compute_variability(df, mean_columns)
    
    # Compute derived features
    if config.get('compute_derived', True):
        df = compute_derived_features(df, config)
    
    tqdm.write("\n✓ Feature engineering complete!")
    tqdm.write(f"  Original columns: {len(df.columns) - len([c for c in df.columns if '_mean' in c or '_std' in c or c in ['map', 'mopp_od', 'mopp_os', 'sopp_od', 'sopp_os', 'dopp_od', 'dopp_os']])}")
    tqdm.write(f"  New feature columns: {len([c for c in df.columns if '_mean' in c or '_std' in c or c in ['map', 'mopp_od', 'mopp_os', 'sopp_od', 'sopp_os', 'dopp_od', 'dopp_os']])}")
    tqdm.write(f"  Total columns: {len(df.columns)}\n")
    
    return df

