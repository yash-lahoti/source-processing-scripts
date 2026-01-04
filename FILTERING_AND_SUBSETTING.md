# Filtering and Subsetting Guide

## Overview

The patient aggregator now supports configurable filtering to create subsets of patients based on data completeness criteria. This is especially useful for EDA (Exploratory Data Analysis) where you want to focus on patients with complete measurement data.

## Features

### 1. Skip Aggregation if File Exists
- If `aggregated_patients.csv` already exists, aggregation is automatically skipped
- Saves time when you only need to regenerate visualizations or subsets
- Use `force=True` in the API or delete the file to force regeneration

### 2. Configurable Filtering
- Define filtering criteria in `config.yaml`
- Create subsets based on required columns and group requirements
- Automatically saves subset to a separate file
- EDA visualizations use the subset by default

## Configuration

### Basic Setup

Add a `filtering` section to your `config.yaml`:

```yaml
filtering:
  enabled: true  # Set to false to disable filtering
  subset_file: "aggregated_patients_subset.csv"  # Output file for filtered subset
  
  # Required columns: each patient must have at least min_count values
  required_columns:
    bp_systolic:
      min_count: 1  # At least 1 SBP measurement
      description: "Systolic blood pressure"
    bp_diastolic:
      min_count: 1  # At least 1 DBP measurement
      description: "Diastolic blood pressure"
    pulse:
      min_count: 1  # At least 1 HR measurement
      description: "Heart rate / pulse"
  
  # Group requirements: patient must have at least min_count values across the group
  group_requirements:
    - name: "iop_measurements"
      columns: ["od_iop", "os_iop"]  # At least one IOP value (either eye)
      min_count: 1
      description: "At least one IOP measurement from either eye"
```

### Filter Types

#### 1. Required Columns
Each patient must have at least `min_count` values in each specified column.

**Example:**
```yaml
required_columns:
  bp_systolic:
    min_count: 1  # Patient must have at least 1 SBP measurement
  bp_diastolic:
    min_count: 2  # Patient must have at least 2 DBP measurements
```

#### 2. Group Requirements
Patient must have at least `min_count` total values across all columns in the group.

**Example:**
```yaml
group_requirements:
  - name: "iop_measurements"
    columns: ["od_iop", "os_iop"]
    min_count: 1  # At least 1 IOP value from either eye
  - name: "vital_signs"
    columns: ["bp_systolic", "bp_diastolic", "pulse"]
    min_count: 3  # At least 3 total vital sign measurements
```

## Usage

### Command Line

```bash
# Run aggregation and create subset
python run_aggregation.py

# Skip plots (faster for testing)
python run_aggregation.py --no-plots
```

### Programmatic Usage

```python
from patient_aggregator import aggregate_patients, create_subset, save_subset
from patient_aggregator.config_loader import load_config, get_filter_config
import pandas as pd

# Aggregate (will skip if file exists)
aggregate_patients('data/', 'output.csv')

# Load aggregated data
df = pd.read_csv('output.csv')

# Load filter config
config = load_config('config.yaml')
filter_config = get_filter_config(config)

# Create subset
subset_df = create_subset(df, filter_config)

# Save subset
save_subset(subset_df, 'subset.csv')
```

## How It Works

1. **Aggregation**: Processes input files and creates `aggregated_patients.csv`
   - Skips if file already exists (unless `force=True`)

2. **Filtering**: Applies filters from config to create subset
   - Checks required columns
   - Checks group requirements
   - Creates filtered DataFrame

3. **Subset Saving**: Saves subset to `aggregated_patients_subset.csv`
   - Skips if subset file exists and is newer than aggregated file

4. **EDA**: Visualizations use the subset by default
   - Full dataset: `aggregated_patients.csv`
   - Subset for EDA: `aggregated_patients_subset.csv`

## Example Output

```
✓ Aggregated file already exists. Skipping aggregation.

Applying filters to create subset...
  Checking required columns:
    bp_systolic: 45/50 patients (min: 1)
    bp_diastolic: 45/50 patients (min: 1)
    pulse: 48/50 patients (min: 1)
    Group 'iop_measurements': 47/50 patients (min: 1 across 2 columns)

✓ Subset created: 42/50 patients (84.0%)
✓ Subset saved: aggregated_patients_subset.csv (42 patients)

📊 Using subset for EDA: aggregated_patients_subset.csv
   Subset: 42 patients
   Full dataset: 50 patients
```

## Disabling Filtering

To disable filtering and use the full dataset:

```yaml
filtering:
  enabled: false
```

Or simply remove the `filtering` section from `config.yaml`.

## Force Regeneration

To force regeneration of aggregated file:

```python
aggregate_patients('data/', 'output.csv', force=True)
```

Or delete the `aggregated_patients.csv` file and rerun.

## Notes

- Filtering is applied after aggregation
- Subset file is only regenerated if it's older than the aggregated file
- All filtering uses JSON array parsing to count values
- Empty arrays `[]` or missing values are treated as 0 values
- Group requirements count total values across all columns in the group

