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


def get_generate_full_plots(config: Dict[str, Any]) -> bool:
    """Get whether to generate full aggregated plots."""
    return config.get('output', {}).get('generate_full_plots', False)


def get_output_directory(config: Dict[str, Any], base_path: Path = None) -> Path:
    """Get output directory from config, creating it if needed."""
    if base_path is None:
        base_path = Path.cwd()
    
    output_dir = config.get('output', {}).get('directory', 'output')
    output_path = base_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_filter_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get filter configuration from config."""
    return config.get('filtering', {})


def get_feature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get feature engineering configuration from config."""
    return config.get('features', {})


def get_stratified_eda_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get stratified EDA configuration from config."""
    return config.get('stratified_eda', {})


def get_group_order(config: Dict[str, Any]) -> List[str]:
    """Get group order from stratified EDA configuration."""
    stratified_config = config.get('stratified_eda', {})
    return stratified_config.get('group_order', [])


def get_statistical_test_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistical test configuration from config."""
    stratified_config = config.get('stratified_eda', {})
    return stratified_config.get('statistical_tests', {})


def get_large_sample_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get large sample handling configuration from statistical tests config."""
    test_config = get_statistical_test_config(config)
    return {
        'large_sample_threshold': test_config.get('large_sample_threshold', 5000),
        'method': test_config.get('method', 'auto'),
        'max_exact_sample_size': test_config.get('max_exact_sample_size', 5000),
        'sample_down_for_exact': test_config.get('sample_down_for_exact', False),
        'suppress_warnings': test_config.get('suppress_warnings', False)
    }


def get_excluded_groups(config: Dict[str, Any]) -> List[str]:
    """Get list of groups to exclude from statistical analysis."""
    test_config = get_statistical_test_config(config)
    return test_config.get('excluded_groups', ['indeterminate', 'intermediate', 'unspecified'])


def get_features_for_statistics(config: Dict[str, Any]) -> List[str]:
    """Get list of features to include in statistical tests."""
    stratified_config = config.get('stratified_eda', {})
    features = stratified_config.get('features_for_statistics', [])
    
    # If not specified, default to all engineered features
    if not features:
        # Get features from feature config
        feature_config = get_feature_config(config)
        mean_cols = feature_config.get('mean_columns', [])
        derived_features = feature_config.get('derived_features', {})
        
        # Build default list
        features = [f"{col}_mean" for col in mean_cols]
        features.extend([f"{col}_std" for col in mean_cols])
        features.extend(list(derived_features.keys()))
    
    return features


def get_plot_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get plot visualization options from config."""
    stratified_config = config.get('stratified_eda', {})
    return stratified_config.get('plot_options', {})

