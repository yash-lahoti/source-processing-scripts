"""Individual statistical test plot generation module."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from scipy import stats


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for data.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower, upper) bounds
    """
    if len(data) == 0:
        return (np.nan, np.nan)
    
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return (mean - h, mean + h)


def create_individual_test_plot(df: pd.DataFrame, stratification_col: pd.Series,
                               feature: str, test_config: Dict[str, Any],
                               output_dir: Path, plot_options: Dict[str, Any] = None) -> None:
    """
    Create a detailed individual plot for a feature's statistical test.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        feature: Feature name to plot
        test_config: Statistical test configuration
        output_dir: Output directory for plots
    """
    if len(stratification_col) == 0 or feature not in df.columns:
        return
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Prepare groups data
    groups_data = {}
    group_stats = {}
    
    for group in df_strat['_group'].dropna().unique():
        group_data = df_strat[df_strat['_group'] == group][feature].dropna()
        if len(group_data) > 0:
            groups_data[str(group)] = group_data.values
            mean = np.mean(group_data.values)
            std = np.std(group_data.values, ddof=1)
            n = len(group_data.values)
            ci_lower, ci_upper = calculate_confidence_interval(group_data.values)
            
            group_stats[str(group)] = {
                'mean': mean,
                'std': std,
                'n': n,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'data': group_data.values
            }
    
    if len(groups_data) < 2:
        return
    
    # Perform statistical tests
    from .statistics import (perform_statistical_tests, format_pvalue, 
                            cohens_d, eta_squared, epsilon_squared)
    
    excluded_groups = test_config.get('excluded_groups', [])
    test_results = perform_statistical_tests(groups_data, test_config, excluded_groups=[])
    
    if 'error' in test_results:
        return
    
    # Get plot options
    if plot_options is None:
        plot_options = {}
    plot_type = plot_options.get('plot_type', 'combined')
    show_outliers = plot_options.get('show_outliers', True)
    scale_type = plot_options.get('scale_type', 'linear')
    show_individual_points = plot_options.get('show_individual_points', True)
    mean_marker_size = plot_options.get('mean_marker_size', 12)
    enhance_mean_visibility = plot_options.get('enhance_mean_visibility', True)
    iqr_zoom_padding = plot_options.get('iqr_zoom_padding', 0.1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Main plot: Enhanced visualization with multiple plot types
    ax_main = fig.add_subplot(gs[0, :])
    
    # Prepare data for plotting
    plot_data = []
    plot_labels = []
    means = []
    cis_lower = []
    cis_upper = []
    ns = []
    positions = []
    all_data_for_scale = []
    
    group_order = sorted(groups_data.keys())
    for i, group in enumerate(group_order):
        stats = group_stats[group]
        plot_data.append(stats['data'])
        plot_labels.append(f"{group}\n(n={stats['n']})")
        means.append(stats['mean'])
        cis_lower.append(stats['ci_lower'])
        cis_upper.append(stats['ci_upper'])
        ns.append(stats['n'])
        positions.append(i + 1)
        all_data_for_scale.extend(stats['data'])
    
    # Create combined plot based on plot_type
    colors = sns.color_palette("husl", len(plot_labels))
    
    if plot_type in ['violin', 'combined']:
        # Create violin plot
        parts = ax_main.violinplot(plot_data, positions=positions, widths=0.6, 
                                   showmeans=False, showmedians=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
    
    if plot_type in ['box', 'combined']:
        # Create box plot
        bp = ax_main.boxplot(plot_data, labels=plot_labels, patch_artist=True, 
                            positions=positions, widths=0.4 if plot_type == 'combined' else 0.6,
                            showfliers=show_outliers)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # Add strip/swarm plot for individual points if requested
    if show_individual_points and plot_type in ['strip', 'swarm', 'combined']:
        for i, (group, data, pos) in enumerate(zip(group_order, plot_data, positions)):
            # Add jitter for strip plot
            jitter = np.random.normal(0, 0.05, len(data))
            ax_main.scatter(pos + jitter, data, alpha=0.4, s=20, color=colors[i], zorder=5)
    
    # Overlay means with error bars (95% CI) - enhanced visibility
    marker_size = mean_marker_size if not enhance_mean_visibility else mean_marker_size * 1.5
    line_width = 2 if not enhance_mean_visibility else 3
    
    ax_main.errorbar(positions, means, 
                     yerr=[np.array(means) - np.array(cis_lower), 
                           np.array(cis_upper) - np.array(means)],
                     fmt='o', color='red', markersize=marker_size, capsize=8, 
                     capthick=line_width, linewidth=line_width,
                     label='Mean (95% CI)', zorder=15, markeredgecolor='darkred', 
                     markeredgewidth=2)
    
    # Apply scaling
    if scale_type == 'iqr_zoom':
        # Calculate IQR for all data
        q1 = np.percentile(all_data_for_scale, 25)
        q3 = np.percentile(all_data_for_scale, 75)
        iqr = q3 - q1
        y_min = q1 - 1.5 * iqr - iqr_zoom_padding * iqr
        y_max = q3 + 1.5 * iqr + iqr_zoom_padding * iqr
        ax_main.set_ylim(y_min, y_max)
        ax_main.text(0.02, 0.98, 'IQR Zoom View', transform=ax_main.transAxes,
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    elif scale_type == 'log':
        ax_main.set_yscale('log')
    elif scale_type == 'symlog':
        ax_main.set_yscale('symlog')
    
    ax_main.set_ylabel('Value', fontsize=12)
    ax_main.set_title(f'{feature} - Statistical Test Results', fontsize=14, fontweight='bold', pad=20)
    ax_main.grid(True, alpha=0.3, axis='y')
    ax_main.legend(loc='upper right')
    
    # Add statistical annotations
    if test_results.get('pairwise_comparisons'):
        y_max = max([max(data) for data in plot_data])
        y_min, y_max_current = ax_main.get_ylim()
        y_range = y_max_current - y_min
        
        from .stratified_eda import annotate_statistical_significance
        annotate_statistical_significance(ax_main, groups_data, positions, test_config, y_max)
    
    # Summary statistics table
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Group', 'N', 'Mean', 'SD', '95% CI Lower', '95% CI Upper']
    
    for group in group_order:
        stats = group_stats[group]
        table_data.append([
            group,
            int(stats['n']),
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['ci_lower']:.3f}",
            f"{stats['ci_upper']:.3f}"
        ])
    
    table = ax_table.table(cellText=table_data, colLabels=headers,
                           cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax_table.set_title('Group Summary Statistics', fontsize=11, fontweight='bold', pad=10)
    
    # Statistical test results table
    ax_test = fig.add_subplot(gs[2, :])
    ax_test.axis('off')
    
    # Overall test
    test_table_data = []
    if test_results.get('overall_test'):
        overall = test_results['overall_test']
        p_str = format_pvalue(overall['p_value'], test_config.get('significance_levels', {}))
        
        # Calculate effect size
        effect_size = np.nan
        if len(groups_data) == 2:
            group_names = list(groups_data.keys())
            effect_size = cohens_d(groups_data[group_names[0]], groups_data[group_names[1]])
        elif len(groups_data) > 2:
            if 'ANOVA' in overall['test']:
                effect_size = eta_squared(groups_data)
            elif 'Kruskal' in overall['test']:
                try:
                    groups_list = [data for data in groups_data.values() if len(data) > 0]
                    h_stat, _ = stats.kruskal(*groups_list)
                    effect_size = epsilon_squared(groups_data, h_stat)
                except:
                    effect_size = np.nan
        
        effect_str = f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
        effect_label = "Cohen's d" if len(groups_data) == 2 else ("η²" if 'ANOVA' in overall['test'] else "ε²")
        
        test_table_data.append(['Overall Test', overall['test'], p_str, f"{effect_label} = {effect_str}"])
    
    # Pairwise comparisons
    if test_results.get('pairwise_comparisons'):
        for (group1, group2), (p_value, test_name) in test_results['pairwise_comparisons'].items():
            p_str = format_pvalue(p_value, test_config.get('significance_levels', {}))
            
            # Calculate effect size
            effect_size = cohens_d(groups_data[str(group1)], groups_data[str(group2)])
            effect_str = f"{effect_size:.3f}" if not np.isnan(effect_size) else "N/A"
            
            mean1 = group_stats[str(group1)]['mean']
            mean2 = group_stats[str(group2)]['mean']
            diff = mean1 - mean2
            
            test_table_data.append([
                f"{group1} vs {group2}",
                test_name,
                p_str,
                f"d = {effect_str}, diff = {diff:.3f}"
            ])
    
    if test_table_data:
        test_headers = ['Comparison', 'Test', 'P-value', 'Effect Size']
        test_table = ax_test.table(cellText=test_table_data, colLabels=test_headers,
                                  cellLoc='left', loc='center',
                                  bbox=[0, 0, 1, 1])
        test_table.auto_set_font_size(False)
        test_table.set_fontsize(9)
        test_table.scale(1, 1.5)
        
        # Style header row
        for i in range(len(test_headers)):
            test_table[(0, i)].set_facecolor('#70AD47')
            test_table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax_test.set_title('Statistical Test Results', fontsize=11, fontweight='bold', pad=10)
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize feature name for filename
    safe_feature = feature.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_path = output_dir / f"individual_test_{safe_feature}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_individual_test_plots(df: pd.DataFrame, stratification_col: pd.Series,
                                    features: List[str], test_config: Dict[str, Any],
                                    output_dir: Path, plot_options: Dict[str, Any] = None) -> None:
    """
    Create individual test plots for all features.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to plot
        test_config: Statistical test configuration
        output_dir: Output directory for plots
    """
    if len(stratification_col) == 0:
        return
    
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        return
    
    # Create subdirectory for individual plots
    individual_dir = output_dir / "individual_tests"
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    tqdm.write("Creating individual statistical test plots...")
    for feature in tqdm(available_features, desc="Individual plots", leave=False, unit="feature"):
        try:
            create_individual_test_plot(df, stratification_col, feature, test_config, 
                                       individual_dir, plot_options=plot_options)
        except Exception as e:
            tqdm.write(f"  ⚠ Error creating plot for {feature}: {e}")
    
    tqdm.write(f"  ✓ Individual plots saved to: {individual_dir}")

