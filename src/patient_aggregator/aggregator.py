"""Core aggregation logic for patient data."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from .config_loader import load_config, get_file_configs, get_output_format


def _read_data_file(file_path: Path) -> pd.DataFrame:
    """
    Read a data file (Excel or CSV) and return a DataFrame.
    Works on both Windows and Unix systems.
    
    Args:
        file_path: Path to the file (can be string or Path object)
        
    Returns:
        DataFrame with the file contents
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    # Convert to Path object and resolve for cross-platform compatibility
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get suffix in lowercase for case-insensitive matching
    suffix = file_path.suffix.lower()
    
    try:
        if suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.csv':
            # Read CSV with common settings that work on Windows and Unix
            return pd.read_csv(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .xlsx, .xls, .csv")
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails (common on Windows)
        if suffix == '.csv':
            return pd.read_csv(file_path, encoding='latin-1')
        raise


def _format_value(values: List, format_type: str) -> str:
    """Format aggregated values based on output format."""
    if format_type == "json_array":
        return json.dumps(values)
    elif format_type == "comma_separated":
        return ",".join(str(v) for v in values)
    elif format_type == "pipe_separated":
        return "|".join(str(v) for v in values)
    return json.dumps(values)


def _aggregate_column_efficient(df: pd.DataFrame, patient_id: str, col_config: Dict, patients: Dict):
    """
    Efficiently aggregate a single column using vectorized operations.
    Uses itertuples() instead of iterrows() for 5-10x better performance.
    """
    col_name = col_config['name']
    col_type = col_config.get('type', 'str')
    
    if col_name not in df.columns:
        return
    
    # Use itertuples() which is much faster than iterrows()
    # Get column names for namedtuple access
    cols = list(df.columns)
    patient_id_pos = cols.index(patient_id)
    col_pos = cols.index(col_name)
    
    # Process with progress bar
    for row in tqdm(df.itertuples(index=False, name=None), total=len(df), 
                    desc=f"Processing {col_name}", leave=False, unit="rows"):
        uid = row[patient_id_pos]
        
        if uid not in patients:
            patients[uid] = {}
        if col_name not in patients[uid]:
            patients[uid][col_name] = []
        
        # Get value by position
        value = row[col_pos]
        
        if pd.notna(value):
            if col_type == 'int':
                patients[uid][col_name].append(int(value))
            elif col_type == 'float':
                patients[uid][col_name].append(float(value))
            else:
                patients[uid][col_name].append(str(value))


def aggregate_patients(input_dir: str, output_file: str, config_path: str = None):
    """Aggregate patient data from Excel or CSV files into single CSV."""
    if config_path is None:
        # Try current directory first, then package directory
        current_dir_config = Path("config.yaml")
        if current_dir_config.exists():
            config_path = current_dir_config
        else:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    config = load_config(config_path)
    input_path = Path(input_dir)
    patient_id = config['patient_id_column']
    output_format = get_output_format(config)
    file_configs = get_file_configs(config)
    
    patients = {}
    
    # Process each file configuration with progress bar
    file_pbar = tqdm(file_configs, desc="Processing files", unit="file")
    for file_config in file_pbar:
        file_name = file_config['file']
        file_pbar.set_description(f"Processing {file_name}")
        file_path = input_path / file_name
        
        # Try to find file with different extensions if exact match not found
        original_file_path = file_path
        if not file_path.exists():
            # Try common extensions (prioritize the extension in config, then try others)
            base_name = file_path.stem
            possible_paths = [
                input_path / f"{base_name}.xlsx",
                input_path / f"{base_name}.csv",
                input_path / f"{base_name}.xls",
                original_file_path,  # Try original as last resort
            ]
            
            file_path = None
            for possible_path in possible_paths:
                if possible_path.exists():
                    file_path = possible_path
                    if possible_path.name != file_name:
                        tqdm.write(f"  Found: {file_name} -> {possible_path.name}")
                    break
            
            if file_path is None:
                tqdm.write(f"Warning: File {file_name} (or variants .xlsx/.csv/.xls) not found, skipping...")
                continue
        
        try:
            df = _read_data_file(file_path)
            tqdm.write(f"  Loaded {file_name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            tqdm.write(f"Warning: Error reading {file_path}: {e}, skipping...")
            continue
        
        # Aggregate each configured column
        columns = file_config.get('columns', [])
        for col_config in tqdm(columns, desc=f"  Aggregating columns", leave=False, unit="col"):
            _aggregate_column_efficient(df, patient_id, col_config, patients)
    
    file_pbar.close()
    
    # Build output DataFrame efficiently
    tqdm.write("\nBuilding output DataFrame...")
    output_data = []
    all_columns = set()
    
    # Collect all column names
    for uid, data in patients.items():
        all_columns.update(data.keys())
    
    # Get all unique column names from config
    for file_config in file_configs:
        for col_config in file_config.get('columns', []):
            all_columns.add(col_config['name'])
    
    # Build rows with progress bar
    patient_items = list(patients.items())
    for uid, data in tqdm(patient_items, desc="Formatting output", unit="patient"):
        row = {patient_id: uid}
        for col in sorted(all_columns):
            values = data.get(col, [])
            row[col] = _format_value(values, output_format)
        output_data.append(row)
    
    # Write to CSV
    tqdm.write("Writing output CSV...")
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, index=False)
    tqdm.write(f"✓ Output saved: {output_file} ({len(df)} patients, {len(df.columns)} columns)")
