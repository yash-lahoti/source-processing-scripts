"""Stratified EDA module for group-based analysis."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from .visualizer import parse_json_array
from .statistics import perform_statistical_tests, format_pvalue
from .summary_tables import (create_summary_statistics_table, save_summary_table,
                             create_statistical_test_table, save_statistical_test_table)
from .statistical_plots import create_all_individual_test_plots


def get_stratification_column(df: pd.DataFrame, config: Dict[str, Any], 
                              exclude_groups: List[str] = None) -> pd.Series:
    """
    Get stratification column, handling cases with multiple values per patient.
    Applies group ordering from config and filters excluded groups.
    
    Args:
        df: DataFrame with patient data
        config: Stratified EDA configuration
        exclude_groups: List of groups to exclude (applied to final series)
        
    Returns:
        Series with single value per patient (most common value used if multiple)
    """
    strat_col_name = config.get('stratification_column', 'source_severity')
    
    if strat_col_name not in df.columns:
        tqdm.write(f"⚠ Warning: Stratification column '{strat_col_name}' not found")
        return pd.Series()
    
    # Handle multiple values per patient (use most common)
    def get_primary_value(values_str):
        values = parse_json_array(values_str)
        if not values:
            return None
        if len(values) == 1:
            return str(values[0])
        # Use most common value
        value_counts = pd.Series(values).value_counts()
        return str(value_counts.index[0])
    
    strat_series = df[strat_col_name].apply(get_primary_value)
    
    # Filter excluded groups
    if exclude_groups:
        strat_series = strat_series[~strat_series.isin(exclude_groups)]
        if len(strat_series) == 0:
            tqdm.write(f"⚠ Warning: All groups excluded, no data remaining")
            return pd.Series()
    
    # Apply group ordering if specified
    group_order = config.get('group_order', [])
    if group_order:
        # Create ordered categories
        all_groups = strat_series.dropna().unique().tolist()
        # Add any groups not in the order list to the end
        ordered_groups = [g for g in group_order if g in all_groups]
        ordered_groups.extend([g for g in all_groups if g not in ordered_groups])
        
        # Convert to categorical with order - ensure we maintain the order
        strat_series = pd.Categorical(strat_series, categories=ordered_groups, ordered=True)
    
    return strat_series


def create_stratified_distributions(df: pd.DataFrame, stratification_col: pd.Series,
                                   features: List[str], output_dir: Path, group_name: str = None):
    """
    Create distribution plots for features, stratified by group.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to plot
        output_dir: Output directory for plots
        group_name: Optional group name for file naming
    """
    if len(stratification_col) == 0:
        return
    
    # Add stratification column to df
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Get unique groups - preserve categorical order if applicable
    if isinstance(stratification_col, pd.Categorical):
        # Categorical object - use .categories directly to preserve order
        cat_categories = list(stratification_col.categories)
        present_groups = set(df_strat['_group'].dropna().tolist())
        groups = [g for g in cat_categories if g in present_groups]
    elif isinstance(stratification_col.dtype, pd.CategoricalDtype):
        # Series with categorical dtype - use .cat accessor
        cat_categories = stratification_col.cat.categories.tolist()
        present_groups = set(df_strat['_group'].dropna().tolist())
        groups = [g for g in cat_categories if g in present_groups]
    else:
        groups = df_strat['_group'].dropna().unique().tolist()
    
    if len(groups) == 0:
        return
    
    # Filter features that exist
    available_features = [f for f in features if f in df_strat.columns]
    
    if not available_features:
        return
    
    # Create subplots
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        
        # Plot distribution for each group
        for group in groups:
            group_data = df_strat[df_strat['_group'] == group][feature].dropna()
            if len(group_data) > 0:
                ax.hist(group_data, alpha=0.6, label=str(group), bins=20)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    filename = f"stratified_distributions_{group_name}.png" if group_name else "stratified_distributions.png"
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    tqdm.write(f"  ✓ Saved: {output_path.name}")


def annotate_statistical_significance(ax, groups_data: Dict[str, np.ndarray], 
                                      positions: List[float], test_config: Dict[str, Any],
                                      y_max: float):
    """
    Add statistical significance annotations to plots.
    
    Args:
        ax: Matplotlib axes object
        groups_data: Dictionary mapping group names to data arrays
        positions: X positions of groups in the plot
        test_config: Statistical test configuration
        y_max: Maximum y-value for positioning annotations
    """
    if not test_config.get('enabled', False):
        return
    
    from .statistics import perform_statistical_tests, format_pvalue
    
    # Perform statistical tests
    test_results = perform_statistical_tests(groups_data, test_config)
    
    if 'error' in test_results or not test_results.get('pairwise_comparisons'):
        return
    
    display_config = test_config.get('display', {})
    show_asterisks = display_config.get('show_asterisks', True)
    show_numeric = display_config.get('show_numeric', True)
    show_brackets = display_config.get('show_brackets', True)
    significance_levels = test_config.get('significance_levels', {"*": 0.05, "**": 0.01, "***": 0.001})
    
    # Get current y-axis limits to calculate proper spacing
    y_min, y_max_current = ax.get_ylim()
    y_range = y_max_current - y_min
    
    # Calculate spacing to avoid overlap with title
    # Use a percentage of the y-range, ensuring enough space
    y_offset = y_range * 0.08  # Increased from 0.05
    bracket_height = y_range * 0.03  # Increased from 0.02
    text_spacing = y_range * 0.02
    
    group_names = list(groups_data.keys())
    
    # Track bracket positions to avoid overlap for multiple comparisons
    bracket_positions = []
    
    # Annotate pairwise comparisons
    for (group1, group2), (p_value, test_name) in test_results['pairwise_comparisons'].items():
        if np.isnan(p_value):
            continue
        
        try:
            idx1 = group_names.index(group1)
            idx2 = group_names.index(group2)
            pos1 = positions[idx1]
            pos2 = positions[idx2]
            
            # Format p-value
            p_str = format_pvalue(p_value, significance_levels)
            
            # Build annotation text
            annotation_parts = []
            if show_numeric:
                annotation_parts.append(f"p={p_str}")
            elif show_asterisks:
                # Extract just asterisks
                asterisks = ''.join([c for c in p_str if c in ['*', '<', '0', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9']])
                annotation_parts.append(asterisks)
            
            annotation_text = ' '.join(annotation_parts) if annotation_parts else p_str
            
            # Draw bracket if enabled
            if show_brackets:
                # Calculate bracket position, avoiding overlap with previous brackets
                bracket_y = y_max_current + y_offset + (len(bracket_positions) * (bracket_height + text_spacing))
                bracket_positions.append((pos1, pos2, bracket_y))
                
                ax.plot([pos1, pos1], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1.5)
                ax.plot([pos2, pos2], [bracket_y, bracket_y + bracket_height], 'k-', linewidth=1.5)
                ax.plot([pos1, pos2], [bracket_y + bracket_height, bracket_y + bracket_height], 'k-', linewidth=1.5)
                
                # Add text above bracket
                text_x = (pos1 + pos2) / 2
                text_y = bracket_y + bracket_height + text_spacing
                ax.text(text_x, text_y, annotation_text, ha='center', va='bottom', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
                
                # Update y-axis limits to accommodate annotations
                new_y_max = text_y + text_spacing * 2
                ax.set_ylim(y_min, new_y_max)
        except (ValueError, IndexError):
            continue


def create_stratified_boxplots(df: pd.DataFrame, stratification_col: pd.Series,
                               features: List[str], output_dir: Path, test_config: Dict[str, Any] = None):
    """
    Create box plots comparing features across groups with statistical annotations.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to plot
        output_dir: Output directory for plots
        test_config: Statistical test configuration
    """
    if len(stratification_col) == 0:
        return
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Sort by group order (if categorical)
    is_categorical = isinstance(stratification_col.dtype, pd.CategoricalDtype) or isinstance(stratification_col, pd.Categorical)
    if is_categorical:
        df_strat = df_strat.sort_values('_group')
    
    groups = df_strat['_group'].dropna().unique()
    if len(groups) == 0:
        return
    
    # Sort groups according to order (if categorical)
    if isinstance(stratification_col, pd.Categorical):
        # Categorical object - use .categories directly
        cat_categories = list(stratification_col.categories)
        groups = sorted(groups, key=lambda x: cat_categories.index(x) if x in cat_categories else 999)
    elif isinstance(stratification_col.dtype, pd.CategoricalDtype):
        # Series with categorical dtype - use .cat accessor
        groups = sorted(groups, key=lambda x: stratification_col.cat.categories.tolist().index(x) if x in stratification_col.cat.categories else 999)
    
    available_features = [f for f in features if f in df_strat.columns]
    if not available_features:
        return
    
    # Create subplots
    n_features = len(available_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        
        # Prepare data for boxplot
        plot_data = []
        plot_labels = []
        groups_data_dict = {}
        positions = []
        
        for i, group in enumerate(groups):
            group_data = df_strat[df_strat['_group'] == group][feature].dropna()
            if len(group_data) > 0:
                plot_data.append(group_data.values)
                plot_labels.append(str(group))
                groups_data_dict[str(group)] = group_data.values
                positions.append(i + 1)
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            # Color boxes
            colors = sns.color_palette("husl", len(plot_labels))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{feature}', pad=15)  # Add padding to prevent overlap with annotations
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add statistical annotations (excluded groups already filtered)
            if test_config and len(groups_data_dict) >= 2:
                y_max = max([max(data) for data in plot_data])
                # Get excluded groups and filter
                excluded_groups = test_config.get('excluded_groups', [])
                filtered_groups_data = {k: v for k, v in groups_data_dict.items() if k not in excluded_groups}
                if len(filtered_groups_data) >= 2:
                    # Adjust positions for filtered groups
                    filtered_positions = [positions[list(groups_data_dict.keys()).index(k)] 
                                        for k in filtered_groups_data.keys()]
                    annotate_statistical_significance(ax, filtered_groups_data, filtered_positions, test_config, y_max)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for annotations
    
    output_path = output_dir / "stratified_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    tqdm.write(f"  ✓ Saved: {output_path.name}")


def create_feature_comparison_plots(df: pd.DataFrame, stratification_col: pd.Series,
                                   features: List[str], output_dir: Path, test_config: Dict[str, Any] = None,
                                   plot_options: Dict[str, Any] = None):
    """
    Create comparison plots for derived features across groups with statistical annotations.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to plot
        output_dir: Output directory for plots
        test_config: Statistical test configuration
    """
    if len(stratification_col) == 0:
        return
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Sort by group order (if categorical)
    is_categorical = isinstance(stratification_col.dtype, pd.CategoricalDtype) or isinstance(stratification_col, pd.Categorical)
    if is_categorical:
        df_strat = df_strat.sort_values('_group')
    
    groups = df_strat['_group'].dropna().unique()
    if len(groups) == 0:
        return
    
    # Sort groups according to order (if categorical)
    if isinstance(stratification_col, pd.Categorical):
        # Categorical object - use .categories directly
        cat_categories = list(stratification_col.categories)
        groups = sorted(groups, key=lambda x: cat_categories.index(x) if x in cat_categories else 999)
    elif isinstance(stratification_col.dtype, pd.CategoricalDtype):
        # Series with categorical dtype - use .cat accessor
        groups = sorted(groups, key=lambda x: stratification_col.cat.categories.tolist().index(x) if x in stratification_col.cat.categories else 999)
    
    # Focus on derived features (MAP, MOPP, SOPP, DOPP)
    derived_features = [f for f in features if f in df_strat.columns and 
                       f in ['map', 'mopp_od', 'mopp_os', 'sopp_od', 'sopp_os', 'dopp_od', 'dopp_os']]
    
    if not derived_features:
        return
    
    # Create comparison plot
    n_features = len(derived_features)
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(derived_features):
        ax = axes[idx]
        
        # Create box plot
        plot_data = []
        plot_labels = []
        groups_data_dict = {}
        positions = []
        
        for i, group in enumerate(groups):
            group_data = df_strat[df_strat['_group'] == group][feature].dropna()
            if len(group_data) > 0:
                plot_data.append(group_data.values)
                plot_labels.append(str(group))
                groups_data_dict[str(group)] = group_data.values
                positions.append(i + 1)
        
        if plot_data:
            # Get plot options
            if plot_options is None:
                plot_options = {}
            plot_type = plot_options.get('plot_type', 'box')
            show_outliers = plot_options.get('show_outliers', True)
            scale_type = plot_options.get('scale_type', 'linear')
            show_individual_points = plot_options.get('show_individual_points', False)
            enhance_mean_visibility = plot_options.get('enhance_mean_visibility', True)
            
            colors = sns.color_palette("husl", len(plot_labels))
            
            # Calculate means for overlay
            means = [np.mean(data) for data in plot_data]
            stds = [np.std(data, ddof=1) for data in plot_data]
            
            # Create combined plot
            if plot_type in ['violin', 'combined']:
                parts = ax.violinplot(plot_data, positions=positions, widths=0.6, 
                                     showmeans=False, showmedians=True)
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
            
            if plot_type in ['box', 'combined']:
                bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                               positions=positions, widths=0.4 if plot_type == 'combined' else 0.6,
                               showfliers=show_outliers)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Add individual points if requested
            if show_individual_points:
                for i, (data, pos) in enumerate(zip(plot_data, positions)):
                    jitter = np.random.normal(0, 0.05, len(data))
                    ax.scatter(pos + jitter, data, alpha=0.3, s=15, color=colors[i], zorder=5)
            
            # Overlay means with error bars (SEM)
            marker_size = 8 if not enhance_mean_visibility else 10
            ax.errorbar(positions, means, yerr=stds, fmt='o', color='red', 
                       markersize=marker_size, capsize=4, capthick=2, linewidth=2,
                       label='Mean ± SD', zorder=10, markeredgecolor='darkred', markeredgewidth=1.5)
            
            # Apply scaling
            if scale_type == 'iqr_zoom':
                all_data = np.concatenate(plot_data)
                q1 = np.percentile(all_data, 25)
                q3 = np.percentile(all_data, 75)
                iqr = q3 - q1
                padding = plot_options.get('iqr_zoom_padding', 0.1) * iqr
                ax.set_ylim(q1 - 1.5 * iqr - padding, q3 + 1.5 * iqr + padding)
            elif scale_type == 'log':
                ax.set_yscale('log')
            elif scale_type == 'symlog':
                ax.set_yscale('symlog')
            
            ax.set_title(f'{feature}', pad=15)  # Add padding to prevent overlap with annotations
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add statistical annotations (excluded groups already filtered)
            if test_config and len(groups_data_dict) >= 2:
                y_max = max([max(data) for data in plot_data])
                # Get excluded groups and filter
                excluded_groups = test_config.get('excluded_groups', [])
                filtered_groups_data = {k: v for k, v in groups_data_dict.items() if k not in excluded_groups}
                if len(filtered_groups_data) >= 2:
                    # Adjust positions for filtered groups
                    filtered_positions = [positions[list(groups_data_dict.keys()).index(k)] 
                                        for k in filtered_groups_data.keys()]
                    annotate_statistical_significance(ax, filtered_groups_data, filtered_positions, test_config, y_max)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for annotations
    
    output_path = output_dir / "feature_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    tqdm.write(f"  ✓ Saved: {output_path.name}")


def save_statistical_test_results(df: pd.DataFrame, stratification_col: pd.Series,
                                 features: List[str], test_config: Dict[str, Any], output_dir: Path):
    """
    Save detailed statistical test results to a text file.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names tested
        test_config: Statistical test configuration
        output_dir: Output directory for results
    """
    if len(stratification_col) == 0:
        return
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    groups = df_strat['_group'].dropna().unique()
    if len(groups) < 2:
        return
    
    available_features = [f for f in features if f in df_strat.columns]
    
    output_path = output_dir / "statistical_test_results.txt"
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STATISTICAL TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for feature in available_features:
            f.write(f"\n{'='*70}\n")
            f.write(f"Feature: {feature}\n")
            f.write(f"{'='*70}\n\n")
            
            # Prepare groups data
            groups_data = {}
            for group in groups:
                group_data = df_strat[df_strat['_group'] == group][feature].dropna()
                if len(group_data) > 0:
                    groups_data[str(group)] = group_data.values
            
            if len(groups_data) < 2:
                f.write("Insufficient groups for statistical testing.\n\n")
                continue
            
            # Get excluded groups from config
            excluded_groups = test_config.get('excluded_groups', [])
            
            # Perform tests (excluded groups already filtered in get_stratification_column)
            test_results = perform_statistical_tests(groups_data, test_config, excluded_groups=[])
            
            if 'error' in test_results:
                f.write(f"Error: {test_results['error']}\n\n")
                continue
            
            # Overall test
            if test_results.get('overall_test'):
                overall = test_results['overall_test']
                f.write(f"Overall Test: {overall['test']}\n")
                f.write(f"  P-value: {overall['p_value']:.6f}\n")
                p_str = format_pvalue(overall['p_value'], test_config.get('significance_levels', {}))
                f.write(f"  Formatted: {p_str}\n\n")
            
            # Pairwise comparisons
            if test_results.get('pairwise_comparisons'):
                f.write("Pairwise Comparisons:\n")
                for (group1, group2), (p_value, test_name) in test_results['pairwise_comparisons'].items():
                    p_str = format_pvalue(p_value, test_config.get('significance_levels', {}))
                    f.write(f"  {group1} vs {group2}: p={p_str} ({test_name})\n")
                f.write("\n")
    
    tqdm.write(f"  ✓ Saved: {output_path.name}")


def create_group_summary_statistics(df: pd.DataFrame, stratification_col: pd.Series,
                                   features: List[str], output_dir: Path):
    """
    Create summary statistics text file for each group.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to summarize
        output_dir: Output directory for statistics
    """
    if len(stratification_col) == 0:
        return
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Get unique groups - preserve categorical order if applicable
    if isinstance(stratification_col, pd.Categorical):
        cat_categories = list(stratification_col.categories)
        groups = [g for g in cat_categories if g in df_strat['_group'].dropna().values]
    elif isinstance(stratification_col.dtype, pd.CategoricalDtype):
        cat_categories = stratification_col.cat.categories.tolist()
        groups = [g for g in cat_categories if g in df_strat['_group'].dropna().values]
    else:
        groups = df_strat['_group'].dropna().unique().tolist()
    
    if len(groups) == 0:
        return
    
    available_features = [f for f in features if f in df_strat.columns]
    
    output_path = output_dir / "group_summary_statistics.txt"
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STRATIFIED SUMMARY STATISTICS BY GROUP\n")
        f.write("="*70 + "\n\n")
        
        for group in sorted(groups):
            group_data = df_strat[df_strat['_group'] == group]
            f.write(f"\n{'='*70}\n")
            f.write(f"GROUP: {group}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Number of patients: {len(group_data)}\n\n")
            
            for feature in available_features:
                values = group_data[feature].dropna()
                if len(values) > 0:
                    f.write(f"{feature}:\n")
                    f.write(f"  Mean: {values.mean():.2f}\n")
                    f.write(f"  Median: {values.median():.2f}\n")
                    f.write(f"  Std Dev: {values.std():.2f}\n")
                    f.write(f"  Min: {values.min():.2f}\n")
                    f.write(f"  Max: {values.max():.2f}\n")
                    f.write(f"  Patients with data: {len(values)}/{len(group_data)}\n\n")
    
    tqdm.write(f"  ✓ Saved: {output_path.name}")


def create_stratified_plots(df: pd.DataFrame, config: Dict[str, Any], output_dir: Path):
    """
    Main function to generate all stratified EDA plots.
    
    Args:
        df: DataFrame with features
        config: Stratified EDA configuration
        output_dir: Output directory for plots
    """
    if not config.get('enabled', False):
        return
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Generating Stratified EDA Plots")
    tqdm.write("="*60 + "\n")
    
    # Get excluded groups from statistical test config
    test_config = config.get('statistical_tests', {})
    excluded_groups = test_config.get('excluded_groups', [])
    
    # Get stratification column (with excluded groups filtered)
    stratification_col = get_stratification_column(df, config, exclude_groups=excluded_groups)
    
    if len(stratification_col) == 0:
        tqdm.write("⚠ No stratification column available, skipping stratified plots")
        return
    
    # Create output subdirectory
    subdir = config.get('output_subdirectory', 'stratified_plots')
    strat_output_dir = output_dir / subdir
    strat_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get features for statistics (all engineered features)
    features_for_statistics = config.get('features_for_statistics', [])
    
    # Get features to plot (for visualization)
    features_to_plot = config.get('features_to_plot', [])
    variability_features = config.get('variability_features', [])
    plot_features = features_to_plot + variability_features
    
    # Create plots
    tqdm.write("Creating stratified distributions...")
    create_stratified_distributions(df, stratification_col, plot_features, strat_output_dir)
    
    tqdm.write("Creating stratified box plots...")
    create_stratified_boxplots(df, stratification_col, plot_features, strat_output_dir, test_config)
    
    # Get plot options
    plot_options = config.get('plot_options', {})
    
    tqdm.write("Creating feature comparison plots...")
    create_feature_comparison_plots(df, stratification_col, features_to_plot, strat_output_dir, 
                                   test_config, plot_options=plot_options)
    
    tqdm.write("Creating group summary statistics...")
    create_group_summary_statistics(df, stratification_col, plot_features, strat_output_dir)
    
    # Generate summary statistics tables
    if config.get('generate_summary_tables', True):
        tqdm.write("Generating summary statistics tables...")
        summary_df = create_summary_statistics_table(
            df, stratification_col, features_for_statistics, strat_output_dir
        )
        table_format = config.get('summary_table_format', 'both')
        save_summary_table(summary_df, strat_output_dir, table_format=table_format)
    
    # Save statistical test results if enabled (using features_for_statistics)
    if test_config.get('enabled', False) and features_for_statistics:
        tqdm.write("Saving statistical test results...")
        save_statistical_test_results(df, stratification_col, features_for_statistics, test_config, strat_output_dir)
        
        # Generate statistical test results table
        tqdm.write("Generating statistical test results table...")
        test_table_df = create_statistical_test_table(
            df, stratification_col, features_for_statistics, test_config
        )
        table_format = config.get('summary_table_format', 'both')
        save_statistical_test_table(test_table_df, strat_output_dir, table_format=table_format)
        
        # Get plot options
        plot_options = config.get('plot_options', {})
        
        # Create individual test plots
        create_all_individual_test_plots(
            df, stratification_col, features_for_statistics, test_config, 
            strat_output_dir, plot_options=plot_options
        )
    
    tqdm.write(f"\n✓ Stratified EDA complete! Plots saved to: {strat_output_dir}\n")

