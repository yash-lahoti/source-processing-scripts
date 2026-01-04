# Running Aggregation Scripts

These scripts allow you to run the patient aggregation directly from your IDE without using terminal commands.

## Quick Start Script: `run_aggregation.py`

The simplest way to run aggregation - just double-click or run from your IDE!

### Features:
- ✅ Automatically finds data directory
- ✅ Scans for Excel files
- ✅ Shows what files were found
- ✅ Runs aggregation automatically
- ✅ Displays summary of results

### How to Use:

1. **Place your Excel files** in one of these directories:
   - `./data/`
   - `./sample_data/`
   - `./input_data/`
   - `./` (project root)

2. **Run the script** from your IDE:
   - Right-click `run_aggregation.py` → "Run"
   - Or use your IDE's run button
   - Or double-click (if configured)

3. **Output** will be saved to `aggregated_patients.csv` in the project root

### Example Output:
```
============================================================
Patient Data Aggregation Script
============================================================

✓ Found data directory: ./sample_data

Found Excel files:
  ✓ cohort.xlsx
  ✓ iop.xlsx
  ✓ diagnosis.xlsx
  ✓ enc.xlsx

Output file: ./aggregated_patients.csv

Starting aggregation...
------------------------------------------------------------
✓ Aggregation complete!

Results saved to: ./aggregated_patients.csv
✓ Total patients aggregated: 4
✓ Columns: 9
```

## Advanced Script: `run_aggregation_with_config.py`

For more control, use this script with command-line arguments.

### Usage:

**Basic (auto-detect everything):**
```python
python run_aggregation_with_config.py
```

**Specify data directory:**
```python
python run_aggregation_with_config.py --data-dir ./my_data
```

**Specify config file:**
```python
python run_aggregation_with_config.py --config custom_config.yaml
```

**Specify output file:**
```python
python run_aggregation_with_config.py --output results.csv
```

**All options:**
```python
python run_aggregation_with_config.py --data-dir ./my_data --config custom_config.yaml --output results.csv
```

### Running from IDE:

Most IDEs allow you to configure script arguments:

**VS Code:**
1. Open `run_aggregation_with_config.py`
2. Click the gear icon next to "Run Python File"
3. Add arguments in `launch.json`:
   ```json
   {
       "name": "Run Aggregation",
       "type": "python",
       "request": "launch",
       "program": "${file}",
       "args": ["--data-dir", "./data", "--output", "results.csv"]
   }
   ```

**PyCharm:**
1. Right-click script → "Modify Run Configuration"
2. Add arguments in "Parameters" field:
   ```
   --data-dir ./data --output results.csv
   ```

## Directory Structure

The scripts will look for Excel files in this order:

1. `./data/` (recommended for your data)
2. `./sample_data/` (sample/test data)
3. `./input_data/`
4. `./` (project root)

## Required Files

The scripts will work with any combination of these files:
- `cohort.xlsx` (optional)
- `iop.xlsx` (required for IOP data)
- `diagnosis.xlsx` (required for diagnosis data)
- `enc.xlsx` (required for encounter/vital signs data)

Missing files will be skipped with a warning.

## Troubleshooting

### "Could not find data directory"
- Make sure Excel files are in one of the directories listed above
- Check that files have `.xlsx` extension

### "No Excel files found"
- Verify files are actually Excel files (`.xlsx` or `.xls`)
- Check file permissions

### Import errors
- Make sure you're running from the project root directory
- The script automatically adds `src/` to the Python path

## Output

The aggregated CSV file will contain:
- One row per patient (grouped by `patient_uid`)
- All aggregated values as JSON arrays
- Columns as specified in `config.yaml`

