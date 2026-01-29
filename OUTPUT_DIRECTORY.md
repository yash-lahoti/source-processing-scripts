# Output Directory Structure

## Overview

All output files are now organized in a dedicated `output/` directory for better project organization and clarity.

## Directory Structure

```
output/
├── aggregated_patients.csv          # Main aggregated dataset
├── aggregated_patients_subset.csv   # Filtered subset for EDA
└── plots/                            # All visualization outputs
    ├── numeric_distributions.png
    ├── numeric_boxplots.png
    ├── categorical_counts.png
    ├── patient_measurement_counts.png
    ├── data_availability_heatmap.png
    ├── patient_cohort_summary.png
    └── summary_statistics.txt
```

## Configuration

The output directory is configured in `config.yaml`:

```yaml
output:
  directory: "output"  # Directory where all output files will be saved
  aggregated_file: "aggregated_patients.csv"  # Main aggregated output file
  format: "json_array"
```

## Customization

### Change Output Directory

Edit `config.yaml`:

```yaml
output:
  directory: "results"  # Change to your preferred directory name
```

### Change File Names

```yaml
output:
  aggregated_file: "my_patients.csv"  # Custom aggregated file name

filtering:
  subset_file: "my_subset.csv"  # Custom subset file name
```

## Benefits

1. **Clean Project Root**: All output files are in one place
2. **Easy to Ignore**: Add `output/` to `.gitignore`
3. **Clear Organization**: Easy to find all results
4. **Version Control**: Keep outputs separate from source code
5. **Easy Cleanup**: Delete `output/` to remove all results

## File Locations

- **Aggregated Data**: `output/aggregated_patients.csv`
- **Filtered Subset**: `output/aggregated_patients_subset.csv`
- **Visualizations**: `output/plots/`
- **Summary Statistics**: `output/plots/summary_statistics.txt`

## Notes

- The output directory is automatically created if it doesn't exist
- All existing functionality works the same, just with organized file locations
- The old `plots/` directory in the root is no longer used
- Old CSV files in the root are ignored (use `output/` instead)



