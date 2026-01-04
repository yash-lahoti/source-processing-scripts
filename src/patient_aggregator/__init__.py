"""Patient data aggregation package."""
from .aggregator import aggregate_patients
from .filter import create_subset, save_subset

__version__ = "0.1.0"
__all__ = ["aggregate_patients", "create_subset", "save_subset"]

