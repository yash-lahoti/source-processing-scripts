"""Core aggregation logic for patient data."""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from .config_loader import load_config, get_file_configs, get_output_format


def _format_value(values: List, format_type: str) -> str:
    """Format aggregated values based on output format."""
    if format_type == "json_array":
        return json.dumps(values)
    elif format_type == "comma_separated":
        return ",".join(str(v) for v in values)
    elif format_type == "pipe_separated":
        return "|".join(str(v) for v in values)
    return json.dumps(values)


def _aggregate_column(df: pd.DataFrame, patient_id: str, col_config: Dict, patients: Dict):
    """Aggregate a single column from a dataframe."""
    col_name = col_config['name']
    col_type = col_config.get('type', 'str')
    agg_type = col_config.get('aggregation', 'list')
    
    for _, row in df.iterrows():
        uid = row[patient_id]
        if uid not in patients:
            patients[uid] = {}
        if col_name not in patients[uid]:
            patients[uid][col_name] = []
        
        value = row.get(col_name)
        if pd.notna(value):
            if col_type == 'int':
                patients[uid][col_name].append(int(value))
            elif col_type == 'float':
                patients[uid][col_name].append(float(value))
            else:
                patients[uid][col_name].append(str(value))


def aggregate_patients(input_dir: str, output_file: str, config_path: str = None):
    """Aggregate patient data from Excel files into single CSV."""
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
    
    # Process each file configuration
    for file_config in file_configs:
        file_name = file_config['file']
        file_path = input_path / file_name
        
        if not file_path.exists():
            print(f"Warning: File {file_name} not found, skipping...")
            continue
        
        df = pd.read_excel(file_path)
        
        # Aggregate each configured column
        for col_config in file_config.get('columns', []):
            _aggregate_column(df, patient_id, col_config, patients)
    
    # Build output DataFrame
    output_data = []
    all_columns = set()
    for uid, data in patients.items():
        all_columns.update(data.keys())
    
    # Get all unique column names from config
    for file_config in file_configs:
        for col_config in file_config.get('columns', []):
            all_columns.add(col_config['name'])
    
    for uid, data in patients.items():
        row = {patient_id: uid}
        for col in sorted(all_columns):
            values = data.get(col, [])
            row[col] = _format_value(values, output_format)
        output_data.append(row)
    
    # Write to CSV
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, index=False)
