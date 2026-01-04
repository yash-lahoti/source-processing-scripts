"""Statistical testing module for group comparisons."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm


def check_normality(data: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Check if data is normally distributed using Shapiro-Wilk test.
    
    Args:
        data: Array of values to test
        alpha: Significance level
        
    Returns:
        True if data appears normal, False otherwise
    """
    if len(data) < 3:
        return False  # Need at least 3 samples for Shapiro-Wilk
    
    try:
        _, p_value = stats.shapiro(data)
        return p_value > alpha
    except:
        return False


def t_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Perform Student's t-test between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Tuple of (p-value, test_name)
    """
    try:
        statistic, p_value = stats.ttest_ind(group1, group2)
        return p_value, "t-test"
    except Exception as e:
        tqdm.write(f"  ⚠ Error in t-test: {e}")
        return np.nan, "t-test"


def mannwhitney_test(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """
    Perform Mann-Whitney U test (non-parametric) between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Tuple of (p-value, test_name)
    """
    try:
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return p_value, "Mann-Whitney U"
    except Exception as e:
        tqdm.write(f"  ⚠ Error in Mann-Whitney test: {e}")
        return np.nan, "Mann-Whitney U"


def anova_test(groups_data: Dict[str, np.ndarray]) -> Tuple[float, str]:
    """
    Perform one-way ANOVA test for multiple groups.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        
    Returns:
        Tuple of (p-value, test_name)
    """
    try:
        groups_list = [data for data in groups_data.values() if len(data) > 0]
        if len(groups_list) < 2:
            return np.nan, "ANOVA"
        
        statistic, p_value = stats.f_oneway(*groups_list)
        return p_value, "ANOVA"
    except Exception as e:
        tqdm.write(f"  ⚠ Error in ANOVA: {e}")
        return np.nan, "ANOVA"


def kruskal_test(groups_data: Dict[str, np.ndarray]) -> Tuple[float, str]:
    """
    Perform Kruskal-Wallis test (non-parametric) for multiple groups.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        
    Returns:
        Tuple of (p-value, test_name)
    """
    try:
        groups_list = [data for data in groups_data.values() if len(data) > 0]
        if len(groups_list) < 2:
            return np.nan, "Kruskal-Wallis"
        
        statistic, p_value = stats.kruskal(*groups_list)
        return p_value, "Kruskal-Wallis"
    except Exception as e:
        tqdm.write(f"  ⚠ Error in Kruskal-Wallis: {e}")
        return np.nan, "Kruskal-Wallis"


def get_pairwise_comparisons(groups_data: Dict[str, np.ndarray], 
                             test_func, apply_correction: bool = True) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """
    Get pairwise comparisons between all groups.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        test_func: Function to use for pairwise tests (t_test or mannwhitney_test)
        apply_correction: Whether to apply Bonferroni correction
        
    Returns:
        Dictionary mapping (group1, group2) tuples to (p-value, test_name) tuples
    """
    results = {}
    group_names = list(groups_data.keys())
    n_comparisons = len(group_names) * (len(group_names) - 1) // 2
    
    for i, name1 in enumerate(group_names):
        for name2 in group_names[i+1:]:
            data1 = groups_data[name1]
            data2 = groups_data[name2]
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            p_value, test_name = test_func(data1, data2)
            
            # Apply Bonferroni correction if requested
            if apply_correction and not np.isnan(p_value):
                corrected_p = min(p_value * n_comparisons, 1.0)
                results[(name1, name2)] = (corrected_p, f"{test_name} (Bonferroni corrected)")
            else:
                results[(name1, name2)] = (p_value, test_name)
    
    return results


def format_pvalue(pvalue: float, significance_levels: Dict[str, float]) -> str:
    """
    Format p-value with asterisks indicating significance.
    
    Args:
        pvalue: P-value to format
        significance_levels: Dictionary mapping asterisk strings to significance levels
        
    Returns:
        Formatted string like "0.023**" or "<0.001***"
    """
    if np.isnan(pvalue):
        return "N/A"
    
    # Determine asterisks
    asterisks = ""
    if pvalue <= significance_levels.get("***", 0.001):
        asterisks = "***"
    elif pvalue <= significance_levels.get("**", 0.01):
        asterisks = "**"
    elif pvalue <= significance_levels.get("*", 0.05):
        asterisks = "*"
    
    # Format numeric value
    if pvalue < 0.001:
        return f"<0.001{asterisks}"
    else:
        return f"{pvalue:.3f}{asterisks}"


def perform_statistical_tests(groups_data: Dict[str, np.ndarray], 
                              test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to perform statistical tests between groups.
    Auto-selects appropriate test based on group count and data distribution.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        test_config: Statistical test configuration
        
    Returns:
        Dictionary with test results including p-values, test names, and pairwise comparisons
    """
    if not test_config.get('enabled', False):
        return {}
    
    # Filter out empty groups
    groups_data = {k: v for k, v in groups_data.items() if len(v) > 0}
    
    if len(groups_data) < 2:
        return {"error": "Need at least 2 groups for statistical testing"}
    
    n_groups = len(groups_data)
    results = {
        "n_groups": n_groups,
        "overall_test": None,
        "pairwise_comparisons": {}
    }
    
    # Check if auto-select is enabled
    auto_select = test_config.get('auto_select', True)
    
    if auto_select:
        # Check normality for all groups
        all_normal = all(check_normality(data) for data in groups_data.values())
        
        if n_groups == 2:
            # Two groups: use t-test if normal, Mann-Whitney otherwise
            if all_normal:
                p_value, test_name = t_test(list(groups_data.values())[0], list(groups_data.values())[1])
                results["overall_test"] = {"p_value": p_value, "test": test_name}
                
                # Pairwise (only one comparison for 2 groups)
                group_names = list(groups_data.keys())
                results["pairwise_comparisons"][tuple(group_names)] = (p_value, test_name)
            else:
                p_value, test_name = mannwhitney_test(list(groups_data.values())[0], list(groups_data.values())[1])
                results["overall_test"] = {"p_value": p_value, "test": test_name}
                
                group_names = list(groups_data.keys())
                results["pairwise_comparisons"][tuple(group_names)] = (p_value, test_name)
        else:
            # Three or more groups: use ANOVA if normal, Kruskal-Wallis otherwise
            if all_normal:
                p_value, test_name = anova_test(groups_data)
                results["overall_test"] = {"p_value": p_value, "test": test_name}
                
                # Pairwise comparisons with t-test
                pairwise = get_pairwise_comparisons(groups_data, t_test, apply_correction=True)
                results["pairwise_comparisons"] = pairwise
            else:
                p_value, test_name = kruskal_test(groups_data)
                results["overall_test"] = {"p_value": p_value, "test": test_name}
                
                # Pairwise comparisons with Mann-Whitney
                pairwise = get_pairwise_comparisons(groups_data, mannwhitney_test, apply_correction=True)
                results["pairwise_comparisons"] = pairwise
    else:
        # Manual test selection (not implemented in this version, use auto_select)
        tqdm.write("  ⚠ Manual test selection not yet implemented, using auto-select")
    
    return results

