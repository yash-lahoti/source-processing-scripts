"""Summary statistics table generation module."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm


def create_summary_statistics_table(df: pd.DataFrame, stratification_col: pd.Series,
                                   features: List[str], output_dir: Path) -> pd.DataFrame:
    """
    Generate summary statistics table (mean, std, min, max, median, n) for all features by group.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to include
        output_dir: Output directory for tables
        
    Returns:
        DataFrame with summary statistics
    """
    if len(stratification_col) == 0:
        return pd.DataFrame()
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    # Filter features that exist
    available_features = [f for f in features if f in df_strat.columns]
    
    if not available_features:
        tqdm.write("  ⚠ No available features for summary table")
        return pd.DataFrame()
    
    # Prepare summary statistics
    summary_rows = []
    
    for feature in tqdm(available_features, desc="Computing summary stats", leave=False, unit="feature"):
        for group in df_strat['_group'].dropna().unique():
            group_data = df_strat[df_strat['_group'] == group][feature].dropna()
            
            if len(group_data) == 0:
                continue
            
            summary_rows.append({
                'Feature': feature,
                'Group': str(group),
                'N': len(group_data),
                'Mean': np.mean(group_data),
                'Std': np.std(group_data),
                'Min': np.min(group_data),
                'Max': np.max(group_data),
                'Median': np.median(group_data),
                'Q25': np.percentile(group_data, 25),
                'Q75': np.percentile(group_data, 75)
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by feature and group
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['Feature', 'Group'])
    
    return summary_df


def save_summary_table(summary_df: pd.DataFrame, output_dir: Path, 
                      table_format: str = 'both') -> None:
    """
    Save summary statistics table in specified format(s).
    
    Args:
        summary_df: DataFrame with summary statistics
        output_dir: Output directory
        table_format: 'csv', 'txt', or 'both'
    """
    if summary_df.empty:
        tqdm.write("  ⚠ No data to save in summary table")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    if table_format in ['csv', 'both']:
        csv_path = output_dir / "summary_statistics_table.csv"
        summary_df.to_csv(csv_path, index=False, float_format='%.3f')
        tqdm.write(f"  ✓ Saved: {csv_path.name}")
    
    # Save formatted text
    if table_format in ['txt', 'both']:
        txt_path = output_dir / "summary_statistics_table.txt"
        with open(txt_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("SUMMARY STATISTICS BY GROUP\n")
            f.write("="*100 + "\n\n")
            
            # Group by feature
            for feature in summary_df['Feature'].unique():
                feature_data = summary_df[summary_df['Feature'] == feature]
                
                f.write(f"\n{'='*100}\n")
                f.write(f"Feature: {feature}\n")
                f.write(f"{'='*100}\n\n")
                
                # Format as table
                f.write(f"{'Group':<20} {'N':<6} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'Q25':<12} {'Q75':<12}\n")
                f.write("-"*100 + "\n")
                
                for _, row in feature_data.iterrows():
                    f.write(f"{row['Group']:<20} {int(row['N']):<6} "
                           f"{row['Mean']:<12.3f} {row['Std']:<12.3f} "
                           f"{row['Min']:<12.3f} {row['Max']:<12.3f} "
                           f"{row['Median']:<12.3f} {row['Q25']:<12.3f} {row['Q75']:<12.3f}\n")
                
                f.write("\n")
        
        tqdm.write(f"  ✓ Saved: {txt_path.name}")


def create_statistical_test_table(df: pd.DataFrame, stratification_col: pd.Series,
                                  features: List[str], test_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate statistical test results table with overall tests and pairwise comparisons.
    
    Args:
        df: DataFrame with features
        stratification_col: Series with group labels
        features: List of feature names to test
        test_config: Statistical test configuration
        
    Returns:
        DataFrame with statistical test results
    """
    if len(stratification_col) == 0:
        return pd.DataFrame()
    
    df_strat = df.copy()
    df_strat['_group'] = stratification_col
    
    groups = df_strat['_group'].dropna().unique()
    if len(groups) < 2:
        return pd.DataFrame()
    
    available_features = [f for f in features if f in df_strat.columns]
    
    if not available_features:
        return pd.DataFrame()
    
    # Import here to avoid circular imports
    from .statistics import (perform_statistical_tests, format_pvalue, 
                            cohens_d, eta_squared, epsilon_squared)
    from scipy import stats
    
    test_rows = []
    excluded_groups = test_config.get('excluded_groups', [])
    
    for feature in tqdm(available_features, desc="Generating test table", leave=False, unit="feature"):
        # Prepare groups data
        groups_data = {}
        group_means = {}
        group_stds = {}
        group_ns = {}
        
        for group in groups:
            group_data = df_strat[df_strat['_group'] == group][feature].dropna()
            if len(group_data) > 0:
                groups_data[str(group)] = group_data.values
                group_means[str(group)] = np.mean(group_data.values)
                group_stds[str(group)] = np.std(group_data.values, ddof=1)
                group_ns[str(group)] = len(group_data.values)
        
        if len(groups_data) < 2:
            continue
        
        # Perform tests
        test_results = perform_statistical_tests(groups_data, test_config, excluded_groups=[])
        
        if 'error' in test_results:
            continue
        
        # Calculate overall effect size
        overall_effect_size = np.nan
        if len(groups_data) == 2:
            # For 2 groups, use Cohen's d
            group_names = list(groups_data.keys())
            overall_effect_size = cohens_d(groups_data[group_names[0]], groups_data[group_names[1]])
        elif len(groups_data) > 2:
            # For 3+ groups, use eta-squared or epsilon-squared
            if 'ANOVA' in str(test_results.get('overall_test', {}).get('test', '')):
                overall_effect_size = eta_squared(groups_data)
            elif 'Kruskal' in str(test_results.get('overall_test', {}).get('test', '')):
                # Need H statistic for epsilon-squared
                try:
                    groups_list = [data for data in groups_data.values() if len(data) > 0]
                    h_stat, _ = stats.kruskal(*groups_list)
                    overall_effect_size = epsilon_squared(groups_data, h_stat)
                except:
                    overall_effect_size = np.nan
        
        # Add overall test result with group means
        if test_results.get('overall_test'):
            overall = test_results['overall_test']
            p_value = overall['p_value']
            test_name = overall['test']
            p_str = format_pvalue(p_value, test_config.get('significance_levels', {}))
            
            # Create summary of group means
            group_means_str = ', '.join([f"{g}: {group_means[g]:.3f}" for g in sorted(group_means.keys())])
            
            test_rows.append({
                'Feature': feature,
                'Test_Type': 'Overall',
                'Group1': 'All',
                'Group2': 'All',
                'Group1_Mean': np.nan,
                'Group2_Mean': np.nan,
                'Mean_Difference': np.nan,
                'Effect_Size': overall_effect_size,
                'P_Value': p_value,
                'Formatted_P_Value': p_str,
                'Test_Name': test_name,
                'Group_Means': group_means_str,
                'Group1_N': np.nan,
                'Group2_N': np.nan
            })
        
        # Add pairwise comparisons with means and effect sizes
        if test_results.get('pairwise_comparisons'):
            for (group1, group2), (p_value, test_name) in test_results['pairwise_comparisons'].items():
                p_str = format_pvalue(p_value, test_config.get('significance_levels', {}))
                
                # Get means and calculate difference
                mean1 = group_means.get(str(group1), np.nan)
                mean2 = group_means.get(str(group2), np.nan)
                mean_diff = mean1 - mean2 if not (np.isnan(mean1) or np.isnan(mean2)) else np.nan
                
                # Calculate Cohen's d for pairwise
                effect_size = np.nan
                if str(group1) in groups_data and str(group2) in groups_data:
                    effect_size = cohens_d(groups_data[str(group1)], groups_data[str(group2)])
                
                test_rows.append({
                    'Feature': feature,
                    'Test_Type': 'Pairwise',
                    'Group1': str(group1),
                    'Group2': str(group2),
                    'Group1_Mean': mean1,
                    'Group2_Mean': mean2,
                    'Mean_Difference': mean_diff,
                    'Effect_Size': effect_size,
                    'P_Value': p_value,
                    'Formatted_P_Value': p_str,
                    'Test_Name': test_name,
                    'Group_Means': np.nan,
                    'Group1_N': group_ns.get(str(group1), np.nan),
                    'Group2_N': group_ns.get(str(group2), np.nan)
                })
    
    test_df = pd.DataFrame(test_rows)
    
    # Sort by feature, then test type (Overall first, then Pairwise), then groups
    if not test_df.empty:
        test_df = test_df.sort_values(['Feature', 'Test_Type', 'Group1', 'Group2'])
    
    return test_df


def save_statistical_test_table(test_df: pd.DataFrame, output_dir: Path,
                                table_format: str = 'both') -> None:
    """
    Save statistical test results table in specified format(s).
    
    Args:
        test_df: DataFrame with statistical test results
        output_dir: Output directory
        table_format: 'csv', 'txt', or 'both'
    """
    if test_df.empty:
        tqdm.write("  ⚠ No data to save in statistical test table")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    if table_format in ['csv', 'both']:
        csv_path = output_dir / "statistical_test_results_table.csv"
        test_df.to_csv(csv_path, index=False, float_format='%.6f')
        tqdm.write(f"  ✓ Saved: {csv_path.name}")
    
    # Save formatted text
    if table_format in ['txt', 'both']:
        txt_path = output_dir / "statistical_test_results_table.txt"
        with open(txt_path, 'w') as f:
            f.write("="*120 + "\n")
            f.write("STATISTICAL TEST RESULTS TABLE\n")
            f.write("="*120 + "\n\n")
            
            # Group by feature
            for feature in test_df['Feature'].unique():
                feature_data = test_df[test_df['Feature'] == feature]
                
                f.write(f"\n{'='*120}\n")
                f.write(f"Feature: {feature}\n")
                f.write(f"{'='*120}\n\n")
                
                # Overall test
                overall_data = feature_data[feature_data['Test_Type'] == 'Overall']
                if not overall_data.empty:
                    f.write("Overall Test:\n")
                    for _, row in overall_data.iterrows():
                        f.write(f"  Test: {row['Test_Name']}\n")
                        f.write(f"  P-value: {row['P_Value']:.6f} ({row['Formatted_P_Value']})\n")
                        if not pd.isna(row.get('Effect_Size', np.nan)):
                            f.write(f"  Effect Size: {row['Effect_Size']:.3f}\n")
                        if 'Group_Means' in row and not pd.isna(row['Group_Means']):
                            f.write(f"  Group Means: {row['Group_Means']}\n")
                    f.write("\n")
                
                # Pairwise comparisons
                pairwise_data = feature_data[feature_data['Test_Type'] == 'Pairwise']
                if not pairwise_data.empty:
                    f.write("Pairwise Comparisons:\n")
                    f.write(f"{'Group1':<15} {'Group2':<15} {'N1':<6} {'N2':<6} {'Mean1':<10} {'Mean2':<10} "
                           f"{'Diff':<10} {'Effect':<10} {'P_Value':<12} {'Formatted':<15} {'Test':<25}\n")
                    f.write("-"*150 + "\n")
                    
                    for _, row in pairwise_data.iterrows():
                        n1 = int(row['Group1_N']) if not pd.isna(row.get('Group1_N', np.nan)) else 'N/A'
                        n2 = int(row['Group2_N']) if not pd.isna(row.get('Group2_N', np.nan)) else 'N/A'
                        mean1 = f"{row['Group1_Mean']:.3f}" if not pd.isna(row.get('Group1_Mean', np.nan)) else 'N/A'
                        mean2 = f"{row['Group2_Mean']:.3f}" if not pd.isna(row.get('Group2_Mean', np.nan)) else 'N/A'
                        diff = f"{row['Mean_Difference']:.3f}" if not pd.isna(row.get('Mean_Difference', np.nan)) else 'N/A'
                        effect = f"{row['Effect_Size']:.3f}" if not pd.isna(row.get('Effect_Size', np.nan)) else 'N/A'
                        
                        f.write(f"{row['Group1']:<15} {row['Group2']:<15} {n1:<6} {n2:<6} {mean1:<10} {mean2:<10} "
                               f"{diff:<10} {effect:<10} {row['P_Value']:<12.6f} {row['Formatted_P_Value']:<15} {row['Test_Name']:<25}\n")
                    f.write("\n")
        
        tqdm.write(f"  ✓ Saved: {txt_path.name}")

