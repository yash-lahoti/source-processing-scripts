# Windows CSV Support Fix

## Changes Made

### 1. File Detection
- Updated `find_data_directory()` to look for both Excel (`.xlsx`, `.xls`) and CSV (`.csv`) files
- Updated `scan_data_directory()` to scan for all supported formats
- Made file matching case-insensitive for Windows compatibility

### 2. File Reading
- Enhanced `_read_data_file()` function with:
  - Cross-platform path handling using `Path.resolve()`
  - UTF-8 encoding for CSV files (with fallback to latin-1 for Windows)
  - Better error messages

### 3. Error Messages
- Updated all error messages to mention both Excel and CSV formats
- More helpful error messages that guide users

## How It Works

The system now:
1. **Searches for data files** in common directories (`data/`, `sample_data/`, `input_data/`, or current directory)
2. **Detects file format** automatically based on extension
3. **Tries alternative extensions** if exact match not found:
   - If config says `iop.xlsx` but only `iop.csv` exists, it will use the CSV file
4. **Works on Windows** with proper path handling and encoding

## Testing on Windows

To test on Windows:

1. **Create CSV files** (if you only have CSV):
   ```python
   import pandas as pd
   pd.read_excel('data.xlsx').to_csv('data.csv', index=False)
   ```

2. **Run the script**:
   ```cmd
   python run_aggregation.py
   ```

3. **The script will automatically**:
   - Find CSV files if Excel files aren't available
   - Read them correctly
   - Process them the same way as Excel files

## Supported File Formats

- ✅ `.xlsx` (Excel 2007+)
- ✅ `.xls` (Excel 97-2003)
- ✅ `.csv` (Comma-separated values)

All formats produce identical results!



