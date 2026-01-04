"""Patient data aggregation package."""
from .aggregator import aggregate_patients
from .filter import create_subset, save_subset
from .features import engineer_features
from .stratified_eda import create_stratified_plots

__version__ = "0.1.0"
__all__ = ["aggregate_patients", "create_subset", "save_subset", "engineer_features", "create_stratified_plots"]

