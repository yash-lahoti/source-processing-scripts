"""Data visualization module for aggregated patient data."""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm


def parse_json_array(value: str) -> List[Any]:
    """Parse JSON array string into Python list."""
    if pd.isna(value) or value == '' or value == '[]':
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def extract_numeric_values(df: pd.DataFrame, column: str) -> List[float]:
    """Extract and flatten all numeric values from a column containing JSON arrays (optimized)."""
    # Use vectorized apply instead of loop
    all_arrays = df[column].apply(parse_json_array)
    all_values = []
    for arr in all_arrays:
        all_values.extend([float(v) for v in arr if isinstance(v, (int, float))])
    return all_values


def extract_categorical_values(df: pd.DataFrame, column: str) -> List[str]:
    """Extract and flatten all categorical values from a column containing JSON arrays (optimized)."""
    # Use vectorized apply instead of loop
    all_arrays = df[column].apply(parse_json_array)
    all_values = []
    for arr in all_arrays:
        all_values.extend([str(v) for v in arr if isinstance(v, str)])
    return all_values


def create_numeric_distributions(df: pd.DataFrame, numeric_columns: List[str], output_dir: Path):
    """Create distribution plots for numeric columns."""
    if not numeric_columns:
        return
    
    # Calculate number of rows/cols for subplots
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_columns):
        if idx >= len(axes):
            break
        
        values = extract_numeric_values(df, col)
        if not values:
            axes[idx].text(0.5, 0.5, f'No data for {col}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            continue
        
        # Create histogram with KDE
        sns.histplot(values, kde=True, ax=axes[idx], bins=20, color='steelblue')
        axes[idx].set_title(f'{col}\n(n={len(values)}, mean={np.mean(values):.1f})', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(numeric_columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'numeric_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_box_plots(df: pd.DataFrame, numeric_columns: List[str], output_dir: Path):
    """Create box plots comparing numeric distributions."""
    if not numeric_columns:
        return
    
    # Prepare data for box plot
    plot_data = []
    for col in numeric_columns:
        values = extract_numeric_values(df, col)
        for val in values:
            plot_data.append({'Variable': col, 'Value': val})
    
    if not plot_data:
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x='Variable', y='Value', hue='Variable', ax=ax, palette='Set2', legend=False)
    ax.set_title('Distribution Comparison: Numeric Variables', fontsize=14, fontweight='bold')
    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / 'numeric_boxplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_categorical_counts(df: pd.DataFrame, categorical_columns: List[str], output_dir: Path):
    """Create bar charts for categorical value counts."""
    if not categorical_columns:
        return
    
    n_cols = min(2, len(categorical_columns))
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(categorical_columns):
        if idx >= len(axes):
            break
        
        values = extract_categorical_values(df, col)
        if not values:
            axes[idx].text(0.5, 0.5, f'No data for {col}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            continue
        
        # Count unique values
        value_counts = pd.Series(values).value_counts().head(20)  # Top 20
        
        # Create bar plot
        bars = axes[idx].barh(range(len(value_counts)), value_counts.values, color='coral')
        axes[idx].set_yticks(range(len(value_counts)))
        axes[idx].set_yticklabels(value_counts.index, fontsize=9)
        axes[idx].set_xlabel('Count', fontsize=11)
        axes[idx].set_title(f'{col}\n(Total unique values: {len(set(values))})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, value_counts.values)):
            axes[idx].text(count, i, f' {count}', va='center', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(categorical_columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'categorical_counts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_summary_statistics(df: pd.DataFrame, numeric_columns: List[str], 
                              categorical_columns: List[str], output_dir: Path):
    """Create a text file with comprehensive summary statistics."""
    output_path = output_dir / 'summary_statistics.txt'
    
    # Find patient ID column
    patient_id_col = None
    for col in df.columns:
        if col.lower() in ['patient_uid', 'patient_id', 'id']:
            patient_id_col = col
            break
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("PATIENT DATA AGGREGATION - COMPREHENSIVE SUMMARY STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Patients: {len(df)}\n\n")
        
        # Patient-level measurement statistics
        if numeric_columns and patient_id_col:
            f.write("PATIENT-LEVEL MEASUREMENT STATISTICS\n")
            f.write("-" * 70 + "\n\n")
            
            for col in numeric_columns:
                counts = count_measurements_per_patient(df, col)
                if len(counts) > 0:
                    patients_with_data = (counts > 0).sum()
                    patients_without_data = (counts == 0).sum()
                    
                    f.write(f"{col}:\n")
                    f.write(f"  Patients with measurements: {patients_with_data} ({patients_with_data/len(df)*100:.1f}%)\n")
                    f.write(f"  Patients without measurements: {patients_without_data} ({patients_without_data/len(df)*100:.1f}%)\n")
                    f.write(f"  Mean measurements per patient: {counts.mean():.2f}\n")
                    f.write(f"  Median measurements per patient: {counts.median():.1f}\n")
                    f.write(f"  Min measurements per patient: {counts.min()}\n")
                    f.write(f"  Max measurements per patient: {counts.max()}\n")
                    
                    # Distribution of measurement counts
                    f.write(f"  Distribution:\n")
                    for count_val in sorted(counts.unique()):
                        num_patients = (counts == count_val).sum()
                        f.write(f"    {count_val} measurement(s): {num_patients} patients ({num_patients/len(df)*100:.1f}%)\n")
                    f.write("\n")
        
        # Data completeness
        if patient_id_col:
            f.write("DATA COMPLETENESS ANALYSIS\n")
            f.write("-" * 70 + "\n\n")
            
            all_columns = numeric_columns + categorical_columns
            completeness_data = []
            
            for col in all_columns:
                counts = count_measurements_per_patient(df, col)
                patients_with_data = (counts > 0).sum() if len(counts) > 0 else 0
                completeness_pct = (patients_with_data / len(df) * 100) if len(df) > 0 else 0
                completeness_data.append({
                    'Column': col,
                    'Patients with Data': patients_with_data,
                    'Completeness %': completeness_pct
                })
            
            completeness_df = pd.DataFrame(completeness_data)
            for _, row in completeness_df.iterrows():
                f.write(f"{row['Column']}: {row['Patients with Data']}/{len(df)} patients ({row['Completeness %']:.1f}%)\n")
            
            f.write(f"\nPatients with complete data (all measurement types): ")
            if all_columns:
                # Vectorized check for complete data
                has_data = pd.DataFrame({
                    col: df[col].apply(lambda x: len(parse_json_array(x)) > 0)
                    for col in all_columns
                })
                complete_patients = has_data.all(axis=1).sum()
                f.write(f"{complete_patients}/{len(df)} ({complete_patients/len(df)*100:.1f}%)\n\n")
            else:
                f.write("N/A\n\n")
        
        # Numeric statistics
        if numeric_columns:
            f.write("NUMERIC VARIABLES\n")
            f.write("-" * 60 + "\n")
            for col in numeric_columns:
                values = extract_numeric_values(df, col)
                if values:
                    f.write(f"\n{col}:\n")
                    f.write(f"  Total measurements: {len(values)}\n")
                    f.write(f"  Mean: {np.mean(values):.2f}\n")
                    f.write(f"  Median: {np.median(values):.2f}\n")
                    f.write(f"  Std Dev: {np.std(values):.2f}\n")
                    f.write(f"  Min: {np.min(values):.2f}\n")
                    f.write(f"  Max: {np.max(values):.2f}\n")
                    f.write(f"  Patients with data: {sum(1 for v in df[col] if parse_json_array(v))}\n")
            f.write("\n")
        
        # Categorical statistics
        if categorical_columns:
            f.write("CATEGORICAL VARIABLES\n")
            f.write("-" * 60 + "\n")
            for col in categorical_columns:
                values = extract_categorical_values(df, col)
                if values:
                    unique_values = set(values)
                    value_counts = pd.Series(values).value_counts()
                    f.write(f"\n{col}:\n")
                    f.write(f"  Total values: {len(values)}\n")
                    f.write(f"  Unique values: {len(unique_values)}\n")
                    f.write(f"  Patients with data: {sum(1 for v in df[col] if parse_json_array(v))}\n")
                    f.write(f"  Top 5 most common:\n")
                    for val, count in value_counts.head(5).items():
                        f.write(f"    - {val}: {count}\n")
            f.write("\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  ✓ Saved: {output_path}")


def count_measurements_per_patient(df: pd.DataFrame, column: str) -> pd.Series:
    """Count number of measurements per patient for a given column (optimized)."""
    patient_id_col = None
    for col in df.columns:
        if col.lower() in ['patient_uid', 'patient_id', 'id']:
            patient_id_col = col
            break
    
    if patient_id_col is None:
        return pd.Series()
    
    # Vectorized operation: apply parse_json_array to entire column
    counts = df[column].apply(lambda x: len(parse_json_array(x)))
    
    return pd.Series(counts.values, index=df[patient_id_col])


def create_measurement_count_distribution(df: pd.DataFrame, numeric_columns: List[str], output_dir: Path):
    """Create histograms showing distribution of measurement counts per patient."""
    if not numeric_columns:
        return
    
    # Find patient ID column
    patient_id_col = None
    for col in df.columns:
        if col.lower() in ['patient_uid', 'patient_id', 'id']:
            patient_id_col = col
            break
    
    if patient_id_col is None:
        return
    
    n_cols = min(3, len(numeric_columns))
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(numeric_columns):
        if idx >= len(axes):
            break
        
        counts = count_measurements_per_patient(df, col)
        if len(counts) == 0:
            axes[idx].text(0.5, 0.5, f'No data for {col}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            continue
        
        # Create histogram
        max_count = counts.max()
        bins = range(0, min(max_count + 2, 11))  # 0-10+ bins
        axes[idx].hist(counts, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel('Number of Measurements per Patient', fontsize=11)
        axes[idx].set_ylabel('Number of Patients', fontsize=11)
        axes[idx].set_title(f'{col}\n(Mean: {counts.mean():.1f}, Median: {counts.median():.1f})', 
                           fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        hist_counts, _ = np.histogram(counts, bins=bins)
        for i, count in enumerate(hist_counts):
            if count > 0:
                axes[idx].text(i, count, f' {int(count)}', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(numeric_columns), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'patient_measurement_counts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_data_availability_heatmap(df: pd.DataFrame, numeric_columns: List[str], 
                                     categorical_columns: List[str], output_dir: Path):
    """Create heatmap showing data availability across patients."""
    # Find patient ID column
    patient_id_col = None
    for col in df.columns:
        if col.lower() in ['patient_uid', 'patient_id', 'id']:
            patient_id_col = col
            break
    
    if patient_id_col is None:
        return
    
    # Create availability matrix
    all_columns = numeric_columns + categorical_columns
    if not all_columns:
        return
    
    availability_data = []
    patient_ids = df[patient_id_col].tolist()
    
    # Limit to first 50 patients for readability
    max_patients = min(50, len(patient_ids))
    patient_ids = patient_ids[:max_patients]
    
    for pid in patient_ids:
        row = df[df[patient_id_col] == pid].iloc[0]
        availability_row = {}
        for col in all_columns:
            values = parse_json_array(row[col])
            availability_row[col] = 1 if len(values) > 0 else 0
        availability_data.append(availability_row)
    
    availability_df = pd.DataFrame(availability_data, index=[f"P{i+1}" for i in range(len(patient_ids))])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(all_columns) * 0.8), max(6, len(patient_ids) * 0.15)))
    sns.heatmap(availability_df.T, annot=False, cmap=['lightcoral', 'lightgreen'], 
                cbar_kws={'label': 'Data Available'}, ax=ax, fmt='d', linewidths=0.5)
    ax.set_title(f'Data Availability Heatmap (First {max_patients} Patients)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Patient', fontsize=12)
    ax.set_ylabel('Measurement Type', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = output_dir / 'data_availability_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def create_patient_cohort_summary(df: pd.DataFrame, categorical_columns: List[str], output_dir: Path):
    """Create summary plots of patient cohort characteristics."""
    if not categorical_columns:
        return
    
    # Find patient ID column
    patient_id_col = None
    for col in df.columns:
        if col.lower() in ['patient_uid', 'patient_id', 'id']:
            patient_id_col = col
            break
    
    if patient_id_col is None:
        return
    
    # Create subplots
    n_plots = min(3, len(categorical_columns))
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    for idx, col in enumerate(categorical_columns[:n_plots]):
        values = extract_categorical_values(df, col)
        if not values:
            axes[idx].text(0.5, 0.5, f'No data for {col}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            continue
        
        # Count unique values
        value_counts = pd.Series(values).value_counts()
        
        # Create pie chart for top categories
        top_n = min(5, len(value_counts))
        top_counts = value_counts.head(top_n)
        other_count = value_counts.iloc[top_n:].sum() if len(value_counts) > top_n else 0
        
        if other_count > 0:
            plot_data = pd.concat([top_counts, pd.Series([other_count], index=['Other'])])
        else:
            plot_data = top_counts
        
        colors = sns.color_palette("husl", len(plot_data))
        wedges, texts, autotexts = axes[idx].pie(plot_data.values, labels=plot_data.index, 
                                                  autopct='%1.1f%%', colors=colors, startangle=90)
        axes[idx].set_title(f'{col}\nDistribution', fontsize=12, fontweight='bold')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # Hide unused subplots
    for idx in range(len(categorical_columns[:n_plots]), n_plots):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'patient_cohort_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def visualize_aggregated_data(csv_path: str, output_dir: str = None):
    """
    Main function to create visualizations from aggregated CSV file.
    
    Args:
        csv_path: Path to the aggregated CSV file
        output_dir: Directory to save plots (default: plots/ in same directory as CSV)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"⚠ Warning: CSV file not found: {csv_path}")
        return False
    
    # Set output directory
    if output_dir is None:
        output_dir = csv_path.parent / 'plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print("Generating Data Visualizations")
    print(f"{'=' * 60}")
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_dir}\n")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        
        # Identify column types
        numeric_columns = []
        categorical_columns = []
        
        # Common numeric column names
        numeric_keywords = ['iop', 'bp_', 'pulse', 'pressure', 'rate', 'count']
        # Common categorical column names
        categorical_keywords = ['severity', 'code', 'desc', 'category', 'type', 'status']
        
        for col in df.columns:
            if col.lower() in ['patient_uid', 'patient_id', 'id']:
                continue
            
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in numeric_keywords):
                numeric_columns.append(col)
            elif any(keyword in col_lower for keyword in categorical_keywords):
                categorical_columns.append(col)
            else:
                # Try to infer: check first non-empty value
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    first_val = parse_json_array(str(sample_values.iloc[0]))
                    if first_val and isinstance(first_val[0], (int, float)):
                        numeric_columns.append(col)
                    else:
                        categorical_columns.append(col)
        
        print(f"Detected {len(numeric_columns)} numeric columns: {numeric_columns}")
        print(f"Detected {len(categorical_columns)} categorical columns: {categorical_columns}\n")
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create visualizations with progress tracking
        viz_tasks = []
        if numeric_columns:
            viz_tasks.append(("Creating numeric distribution plots...", 
                            lambda: create_numeric_distributions(df, numeric_columns, output_dir)))
            viz_tasks.append(("Creating box plots...", 
                            lambda: create_box_plots(df, numeric_columns, output_dir)))
            viz_tasks.append(("Creating patient measurement count distributions...", 
                            lambda: create_measurement_count_distribution(df, numeric_columns, output_dir)))
        
        if categorical_columns:
            viz_tasks.append(("Creating categorical count plots...", 
                            lambda: create_categorical_counts(df, categorical_columns, output_dir)))
            viz_tasks.append(("Creating patient cohort summary...", 
                            lambda: create_patient_cohort_summary(df, categorical_columns, output_dir)))
        
        viz_tasks.append(("Creating data availability heatmap...", 
                        lambda: create_data_availability_heatmap(df, numeric_columns, categorical_columns, output_dir)))
        viz_tasks.append(("Generating summary statistics...", 
                        lambda: create_summary_statistics(df, numeric_columns, categorical_columns, output_dir)))
        
        for desc, func in tqdm(viz_tasks, desc="Generating visualizations", unit="plot"):
            func()
        
        print(f"\n{'=' * 60}")
        print("✓ Visualization complete!")
        print(f"All plots saved to: {output_dir}")
        print(f"{'=' * 60}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

