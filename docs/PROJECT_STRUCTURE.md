# Project Structure

This document describes the organization of the patient-aggregator package.

## Directory Structure

```
SOURCE/
├── LICENSE                    # MIT License
├── README.md                   # Main package documentation
├── pyproject.toml              # Package configuration
├── MANIFEST.in                 # Files to include in distribution
├── config.yaml                 # Default configuration file
│
├── src/                        # Source code (Python package)
│   └── patient_aggregator/
│       ├── __init__.py
│       ├── aggregator.py       # Core aggregation logic
│       ├── cli.py              # Command-line interface
│       └── config_loader.py    # Configuration loader
│
├── tests/                      # Test directory (placeholder)
│
├── docs/                       # Additional documentation
│   ├── README.md               # Documentation index
│   ├── DEPLOYMENT.md           # PyPI deployment guide
│   ├── QUICK_START.md          # Quick deployment reference
│   ├── TUTORIAL_CHECKLIST.md   # Packaging tutorial checklist
│   └── file_structure.md       # Excel file structure documentation
│
├── scripts/                    # Utility scripts
│   └── create_sample_data.py  # Generate sample Excel files
│
├── sample_data/                # Sample Excel files for testing
│   ├── cohort.xlsx
│   ├── iop.xlsx
│   ├── diagnosis.xlsx
│   └── enc.xlsx
│
├── data/                       # Reference data
│   └── image.png               # Original data structure image
│
└── dist/                       # Build output (generated, not committed)
    ├── patient-aggregator-0.1.0.tar.gz
    └── patient-aggregator-0.1.0-py3-none-any.whl
```

## File Organization

### Root Directory
- **Essential files**: `LICENSE`, `README.md`, `pyproject.toml`, `MANIFEST.in`
- **Configuration**: `config.yaml` (default config used by package)
- **Build output**: `dist/` (generated, excluded from git)

### Source Code (`src/`)
- Follows Python packaging tutorial's recommended `src/` layout
- Contains the main package: `patient_aggregator`

### Documentation (`docs/`)
- All additional documentation files
- Keeps root directory clean
- Easy to find and maintain

### Scripts (`scripts/`)
- Utility scripts for development and testing
- Example: `create_sample_data.py` generates test data

### Sample Data (`sample_data/`)
- Example Excel files for testing
- Can be used to verify package functionality

### Tests (`tests/`)
- Placeholder for test files
- Ready for adding unit tests

## Key Files

- **`pyproject.toml`**: Complete package metadata and configuration
- **`config.yaml`**: Default configuration for aggregation
- **`README.md`**: Main documentation (shown on PyPI)
- **`docs/DEPLOYMENT.md`**: Step-by-step PyPI deployment guide

