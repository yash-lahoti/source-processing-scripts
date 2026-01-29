# File Format Support

The patient aggregator supports both **Excel** (`.xlsx`, `.xls`) and **CSV** (`.csv`) file formats for input data files.

## Supported Formats

- **Excel**: `.xlsx`, `.xls`
- **CSV**: `.csv`

## Automatic Format Detection

The system automatically detects the file format based on the file extension. You can specify either format in your `config.yaml`:

```yaml
files:
  - name: "iop"
    file: "iop.xlsx"  # Excel format
    # OR
    file: "iop.csv"   # CSV format
```

## Flexible File Matching

If a file with the exact name specified in the config is not found, the system will automatically try alternative extensions:

1. First tries the exact filename from config
2. If not found, tries `.xlsx`
3. Then tries `.csv`
4. Finally tries `.xls`

**Example:**
- Config specifies: `"iop.xlsx"`
- If `iop.xlsx` doesn't exist, it will try `iop.csv`
- If `iop.csv` exists, it will use that file

## Usage Examples

### All Excel Files
```yaml
files:
  - name: "iop"
    file: "iop.xlsx"
  - name: "diagnosis"
    file: "diagnosis.xlsx"
```

### All CSV Files
```yaml
files:
  - name: "iop"
    file: "iop.csv"
  - name: "diagnosis"
    file: "diagnosis.csv"
```

### Mixed Formats
```yaml
files:
  - name: "iop"
    file: "iop.xlsx"      # Excel
  - name: "diagnosis"
    file: "diagnosis.csv" # CSV
```

The system will automatically handle the different formats seamlessly.

## Converting Between Formats

If you need to convert Excel files to CSV:

```python
import pandas as pd

# Read Excel
df = pd.read_excel('data.xlsx')

# Write CSV
df.to_csv('data.csv', index=False)
```

## Notes

- CSV files are read using `pandas.read_csv()`
- Excel files are read using `pandas.read_excel()` (requires `openpyxl` for `.xlsx`)
- Both formats produce identical aggregation results
- The output is always a CSV file regardless of input format



