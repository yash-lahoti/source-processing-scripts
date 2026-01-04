"""Feature engineering module for patient data."""
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

# Validation thresholds (can be overridden by config)
VALIDATION_THRESHOLDS = {
    'map': {'min': 0, 'max': 250, 'description': 'MAP should be positive and < 250 mmHg'},
    'mopp_od': {'min': -50, 'max': 200, 'description': 'MOPP typically -50 to 200 mmHg'},
    'mopp_os': {'min': -50, 'max': 200, 'description': 'MOPP typically -50 to 200 mmHg'},
    'sopp_od': {'min': -50, 'max': 200, 'description': 'SOPP typically -50 to 200 mmHg'},
    'sopp_os': {'min': -50, 'max': 200, 'description': 'SOPP typically -50 to 200 mmHg'},
    'dopp_od': {'min': -50, 'max': 200, 'description': 'DOPP typically -50 to 200 mmHg'},
    'dopp_os': {'min': -50, 'max': 200, 'description': 'DOPP typically -50 to 200 mmHg'},
}

# Track parse failures for reporting
_parse_failures = {'count': 0, 'examples': []}


def parse_json_array(value: str) -> Tuple[List, bool]:
    """
    Parse a JSON array string into a Python list.
    
    Args:
        value: String value to parse
        
    Returns:
        Tuple of (parsed list, success flag)
    """
    if pd.isna(value) or value == '' or value == '[]':
        return [], True
    
    # Check for common malformed patterns
    malformed_patterns = [
        (r'^\d+,\d+', 'comma-separated without brackets'),
        (r'^\[\d+,\s*\d+$', 'unclosed bracket'),
        (r'^\d+;\d+', 'semicolon-separated'),
    ]
    
    for pattern, desc in malformed_patterns:
        if re.match(pattern, str(value).strip()):
            _parse_failures['count'] += 1
            if len(_parse_failures['examples']) < 5:
                _parse_failures['examples'].append((str(value)[:50], desc))
            return [], False
    
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed, True
        return [parsed], True
    except (json.JSONDecodeError, TypeError) as e:
        _parse_failures['count'] += 1
        if len(_parse_failures['examples']) < 5:
            _parse_failures['examples'].append((str(value)[:50], str(e)))
        return [], False


def _convert_to_numeric(values: List) -> np.ndarray:
    """Convert list of values to numeric array, filtering out invalid values."""
    if not values:
        return np.array([])
    numeric_values = pd.to_numeric(values, errors='coerce')
    return numeric_values[~pd.isna(numeric_values)].values


def _safe_eval_formula(formula: str, df: pd.DataFrame, allowed_columns: List[str]) -> Optional[pd.Series]:
    """
    Safely evaluate a formula using pandas.eval() with column name whitelist.
    
    Args:
        formula: Formula string (e.g., "bp_diastolic_mean + (1/3) * (bp_systolic_mean - bp_diastolic_mean)")
        df: DataFrame containing the columns
        allowed_columns: Whitelist of allowed column names
        
    Returns:
        Series with computed values, or None if evaluation fails
    """
    if not formula:
        return None
    
    # Build a local namespace with only whitelisted columns as variables
    # This is safer than using df["column"] syntax
    local_dict = {}
    for col in allowed_columns:
        if col in df.columns:
            local_dict[col] = df[col]
    
    # Verify all column names in formula are in the whitelist
    # Extract potential column names (simple heuristic: alphanumeric + underscore)
    potential_cols = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula))
    # Filter out Python keywords and operators
    python_keywords = {'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while', 'def', 'class', 'import', 'from', 'as', 'True', 'False', 'None'}
    potential_cols = potential_cols - python_keywords - {'df', 'np', 'pd'}
    
    # Check if all referenced columns are in whitelist
    for col in potential_cols:
        if col not in local_dict:
            # Column not in DataFrame or not whitelisted
            return None
    
    try:
        # Use pandas.eval() which is safer than eval()
        # pandas.eval() supports arithmetic operations on Series
        result = pd.eval(formula, local_dict=local_dict)
        if isinstance(result, pd.Series):
            return result
        return None
    except Exception:
        return None


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
        
        def compute_mean(value):
            """Parse once, convert to numeric, then compute mean."""
            parsed, success = parse_json_array(value)
            if not parsed:
                return np.nan
            numeric_values = _convert_to_numeric(parsed)
            if len(numeric_values) == 0:
                return np.nan
            return np.mean(numeric_values)
        
        df[mean_col] = df[col].apply(compute_mean)
    
    return df


def compute_variability(df: pd.DataFrame, columns: List[str], ddof: int = 1) -> pd.DataFrame:
    """
    Compute standard deviation (variability) for each patient across specified columns.
    Uses sample standard deviation (ddof=1) by default. Only computes if patient has 2+ measurements.
    
    Args:
        df: DataFrame with aggregated patient data
        columns: List of column names to compute variability for
        ddof: Delta degrees of freedom. Default is 1 for sample std. Use 0 for population std.
        
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
            """Parse, convert to numeric, then compute sample std."""
            parsed, _ = parse_json_array(values_str)
            numeric_values = _convert_to_numeric(parsed)
            if len(numeric_values) < 2:
                return np.nan  # Need at least 2 values for std
            return np.std(numeric_values, ddof=ddof)
        
        df[std_col] = df[col].apply(compute_std)
    
    return df


def _check_required_columns(df: pd.DataFrame, required: List[str], feature_name: str) -> bool:
    """Check if required columns exist, log warning if missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        tqdm.write(f"  ⚠ {feature_name}: Missing required columns: {', '.join(missing)}. Skipping.")
        return False
    return True


def _validate_feature_values(df: pd.DataFrame, feature_name: str, thresholds: Dict[str, Any]) -> None:
    """Validate feature values against thresholds and log warnings."""
    if feature_name not in thresholds:
        return
    
    thresh = thresholds[feature_name]
    min_val = thresh.get('min', -np.inf)
    max_val = thresh.get('max', np.inf)
    
    if feature_name not in df.columns:
        return
    
    n_below = (df[feature_name] < min_val).sum()
    n_above = (df[feature_name] > max_val).sum()
    
    if n_below > 0:
        tqdm.write(f"  ⚠ {feature_name}: {n_below} values < {min_val} ({thresh.get('description', '')})")
    if n_above > 0:
        tqdm.write(f"  ⚠ {feature_name}: {n_above} values > {max_val} ({thresh.get('description', '')})")


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
    validation_thresholds = config.get('validation_thresholds', VALIDATION_THRESHOLDS)
    
    if not derived_features:
        return df
    
    tqdm.write("Computing derived features...")
    
    # Get all available column names for formula evaluation whitelist
    available_columns = list(df.columns)
    
    # Process features in dependency order: MAP first (needed for MOPP)
    # Sort to process 'map' first if present
    feature_order = sorted(derived_features.keys(), key=lambda x: (x != 'map', x))
    
    for feat_name in feature_order:
        feat_config = derived_features[feat_name]
        formula = feat_config.get('formula', '')
        description = feat_config.get('description', feat_name)
        
        try:
            # Use formula from config if available, otherwise skip
            if not formula:
                tqdm.write(f"  ⚠ {feat_name}: No formula in config, skipping")
                continue
            
            # Check required columns based on formula
            # Extract column names from formula (simple heuristic)
            formula_lower = formula.lower()
            required_cols = []
            
            if 'bp_diastolic_mean' in formula_lower or 'dbp' in formula_lower:
                required_cols.append('bp_diastolic_mean')
            if 'bp_systolic_mean' in formula_lower or 'sbp' in formula_lower:
                required_cols.append('bp_systolic_mean')
            if 'map' in formula_lower and feat_name != 'map':
                required_cols.append('map')
            if 'od_iop_mean' in formula_lower:
                required_cols.append('od_iop_mean')
            if 'os_iop_mean' in formula_lower:
                required_cols.append('os_iop_mean')
            
            # For MAP, check both BP columns
            if feat_name == 'map':
                required_cols = ['bp_diastolic_mean', 'bp_systolic_mean']
            
            if required_cols and not _check_required_columns(df, required_cols, feat_name):
                continue
            
            # Evaluate formula safely
            result = _safe_eval_formula(formula, df, available_columns)
            
            if result is None:
                tqdm.write(f"  ⚠ {feat_name}: Failed to evaluate formula: {formula}")
                continue
            
            df[feat_name] = result
            
            # Validate values
            _validate_feature_values(df, feat_name, validation_thresholds)
            
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
    
    # Reset parse failure tracking
    global _parse_failures
    _parse_failures = {'count': 0, 'examples': []}
    
    # Track original columns for accurate accounting
    original_columns = set(df.columns)
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Feature Engineering Pipeline")
    tqdm.write("="*60 + "\n")
    
    # Apply data cleaning if enabled
    try:
        from .data_cleaning import clean_feature_data
    except ImportError:
        from patient_aggregator.data_cleaning import clean_feature_data
    
    df, cleaning_stats = clean_feature_data(df, config)
    
    # Compute means
    if config.get('compute_means', True):
        mean_columns = config.get('mean_columns', [])
        if mean_columns:
            df = compute_patient_means(df, mean_columns)
    
    # Compute variability
    if config.get('compute_variability', True):
        mean_columns = config.get('mean_columns', [])
        if mean_columns:
            ddof = config.get('std_ddof', 1)  # Default to sample std
            df = compute_variability(df, mean_columns, ddof=ddof)
    
    # Compute derived features
    if config.get('compute_derived', True):
        df = compute_derived_features(df, config)
    
    # Report parse failures if any
    if _parse_failures['count'] > 0:
        tqdm.write(f"\n⚠ JSON parse failures: {_parse_failures['count']} total")
        if _parse_failures['examples']:
            tqdm.write("  Example failures:")
            for example, reason in _parse_failures['examples'][:3]:
                tqdm.write(f"    - '{example}...' ({reason})")
    
    # Accurate column accounting
    new_columns = set(df.columns) - original_columns
    tqdm.write("\n✓ Feature engineering complete!")
    tqdm.write(f"  Original columns: {len(original_columns)}")
    tqdm.write(f"  New feature columns: {len(new_columns)}")
    tqdm.write(f"  Total columns: {len(df.columns)}")
    if new_columns:
        tqdm.write(f"  New columns: {', '.join(sorted(new_columns))}")
    tqdm.write("")
    
    return df

