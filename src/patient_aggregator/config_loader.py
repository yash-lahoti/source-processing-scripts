"""Configuration loader for patient aggregation."""
import yaml
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required = ['patient_id_column', 'files']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    return config


def get_file_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract file configurations from main config."""
    return config.get('files', [])


def get_output_format(config: Dict[str, Any]) -> str:
    """Get output format from config."""
    return config.get('output', {}).get('format', 'json_array')

