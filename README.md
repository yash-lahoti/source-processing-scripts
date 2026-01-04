# Patient Aggregator

A configurable Python package to aggregate patient data from multiple Excel files into a single CSV file.

## Installation

### From PyPI (once published)

```bash
pip install patient-aggregator
```

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/patient-aggregator.git
cd patient-aggregator

# Install the package
pip install .

# Or install in development mode
pip install -e .
```

## Configuration

The package uses a YAML configuration file (`config.yaml`) to specify which files and columns to aggregate. This makes the system modular and easily configurable.

### Config File Structure

```yaml
# Patient identifier column name (used to group records)
patient_id_column: "patient_uid"

# Input files configuration
files:
  - name: "iop"
    file: "iop.xlsx"
    columns:
      - name: "od_iop"
        type: "int"
        aggregation: "list"
      - name: "os_iop"
        type: "int"
        aggregation: "list"
  
  - name: "diagnosis"
    file: "diagnosis.xlsx"
    columns:
      - name: "source_severity"
        type: "str"
        aggregation: "list"
      - name: "code"
        type: "str"
        aggregation: "list"

# Output configuration
output:
  format: "json_array"  # Options: json_array, comma_separated, pipe_separated
```

### Config Options

- **patient_id_column**: Column name used to group/identify patients (default: "patient_uid")
- **files**: List of file configurations
  - **name**: Logical name for the file
  - **file**: Excel filename
  - **columns**: List of columns to aggregate
    - **name**: Column name in the Excel file
    - **type**: Data type ("int", "float", "str")
    - **aggregation**: Aggregation method (currently supports "list")
- **output.format**: Output format for aggregated values
  - `json_array`: JSON array format (default)
  - `comma_separated`: Comma-separated values
  - `pipe_separated`: Pipe-separated values

## Usage

### Command Line

After installation, use the `aggregate-patients` command:

```bash
# Use default config.yaml in project root
aggregate-patients --input-dir ./data --output aggregated_patients.csv

# Specify custom config file
aggregate-patients --input-dir ./data --output aggregated_patients.csv --config custom_config.yaml
```

### Python API

```python
from patient_aggregator import aggregate_patients

# Use default config.yaml
aggregate_patients(input_dir="./data", output_file="aggregated_patients.csv")

# Use custom config file
aggregate_patients(input_dir="./data", output_file="aggregated_patients.csv", config_path="custom_config.yaml")
```

## Input Files

The package reads Excel files specified in the configuration file. By default, it expects:

- `cohort.xlsx` - Patient cohort information
- `iop.xlsx` - Intraocular pressure measurements
- `diagnosis.xlsx` - Diagnosis records
- `enc.xlsx` - Patient encounter records

You can modify `config.yaml` to add or remove files and columns as needed.

## Output

The output CSV file contains one row per patient (grouped by `patient_id_column`) with columns specified in the configuration. Multiple values for the same patient are aggregated based on the output format specified in the config.

## Sample Data

Sample Excel files are provided in the `sample_data/` directory for testing.

You can regenerate sample data using the script in the `scripts/` directory:

```bash
python scripts/create_sample_data.py
```

## Additional Documentation

See the `docs/` directory for:
- Deployment guide (`DEPLOYMENT.md`)
- Quick start guide (`QUICK_START.md`)
- Tutorial checklist (`TUTORIAL_CHECKLIST.md`)
- File structure documentation (`file_structure.md`)

