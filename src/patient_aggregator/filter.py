"""Filtering and subsetting functions for aggregated patient data."""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
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


def has_non_empty_values(df: pd.DataFrame, column: str) -> pd.Series:
    """Check if each patient has at least one non-empty value in the column."""
    return df[column].apply(lambda x: len(parse_json_array(x)) > 0)


def create_subset(df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a subset of patients based on filtering criteria.
    
    Args:
        df: Aggregated patient DataFrame
        filter_config: Filter configuration dictionary
        
    Returns:
        Filtered DataFrame
    """
    if not filter_config or not filter_config.get('enabled', False):
        return df
    
    tqdm.write("\nApplying filters to create subset...")
    
    # Start with all patients
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Get required columns from config
    required_columns = filter_config.get('required_columns', {})
    
    if required_columns:
        tqdm.write("  Checking required columns:")
        
        # Check each required column
        for col_name, col_config in required_columns.items():
            if col_name not in df.columns:
                tqdm.write(f"    ⚠ Warning: Column '{col_name}' not found, skipping filter")
                continue
            
            min_count = col_config.get('min_count', 1)
            has_values = has_non_empty_values(df, col_name)
            
            # Count values per patient
            value_counts = df[col_name].apply(lambda x: len(parse_json_array(x)))
            meets_criteria = value_counts >= min_count
            
            initial_count = mask.sum()
            mask = mask & meets_criteria
            filtered_count = mask.sum()
            
            tqdm.write(f"    {col_name}: {filtered_count}/{initial_count} patients (min: {min_count})")
    
    # Apply any group requirements (e.g., "at least one from group")
    group_requirements = filter_config.get('group_requirements', [])
    
    for group in group_requirements:
        group_name = group.get('name', 'unnamed')
        columns = group.get('columns', [])
        min_count = group.get('min_count', 1)
        
        if not columns:
            continue
        
        # Check which columns exist
        existing_cols = [col for col in columns if col in df.columns]
        if not existing_cols:
            tqdm.write(f"    ⚠ Warning: No columns from group '{group_name}' found")
            continue
        
        # Check if patient has at least min_count values across the group
        group_mask = pd.Series([False] * len(df), index=df.index)
        for col in existing_cols:
            has_values = has_non_empty_values(df, col)
            group_mask = group_mask | has_values
        
        # Count total values across group
        total_values = pd.Series([0] * len(df), index=df.index)
        for col in existing_cols:
            value_counts = df[col].apply(lambda x: len(parse_json_array(x)))
            total_values = total_values + value_counts
        
        meets_group_criteria = total_values >= min_count
        
        initial_count = mask.sum()
        mask = mask & meets_group_criteria
        filtered_count = mask.sum()
        
        tqdm.write(f"    Group '{group_name}': {filtered_count}/{initial_count} patients (min: {min_count} across {len(existing_cols)} columns)")
    
    # Apply final mask
    filtered_df = df[mask].copy()
    
    tqdm.write(f"\n✓ Subset created: {len(filtered_df)}/{len(df)} patients ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df


def save_subset(df: pd.DataFrame, output_path: str):
    """Save subset DataFrame to CSV."""
    output_file = Path(output_path)
    df.to_csv(output_file, index=False)
    tqdm.write(f"✓ Subset saved: {output_path} ({len(df)} patients)")



