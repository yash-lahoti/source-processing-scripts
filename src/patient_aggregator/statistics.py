"""Statistical testing module for group comparisons."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warnings_context(suppress: bool = False):
    """Context manager to suppress scipy warnings if configured."""
    if suppress:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', message='.*p-value may not be accurate.*')
            yield
    else:
        yield


def _handle_large_samples(group1: np.ndarray, group2: np.ndarray, 
                          large_sample_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Handle large sample sizes by applying sampling if needed.
    
    Args:
        group1: First group data
        group2: Second group data
        large_sample_config: Configuration for large sample handling
        
    Returns:
        Tuple of (processed_group1, processed_group2, method_to_use)
    """
    method = large_sample_config.get('method', 'auto')
    max_exact = large_sample_config.get('max_exact_sample_size', 5000)
    sample_down = large_sample_config.get('sample_down_for_exact', False)
    
    n1, n2 = len(group1), len(group2)
    total_n = n1 + n2
    
    # Determine method
    if method == 'exact':
        use_method = 'exact'
        if sample_down and (n1 > max_exact or n2 > max_exact):
            # Sample down
            np.random.seed(42)  # For reproducibility
            if n1 > max_exact:
                group1 = np.random.choice(group1, size=max_exact, replace=False)
            if n2 > max_exact:
                group2 = np.random.choice(group2, size=max_exact, replace=False)
            tqdm.write(f"  ⚠ Sampled down to {max_exact} for exact method")
    elif method == 'asymptotic':
        use_method = 'asymptotic'
    else:  # auto
        # Use exact if both groups are small enough
        if n1 <= max_exact and n2 <= max_exact:
            use_method = 'exact'
        else:
            use_method = 'asymptotic'
            if total_n > large_sample_config.get('large_sample_threshold', 5000):
                tqdm.write(f"  ⚠ Large sample size (N={total_n}), using asymptotic method")
    
    return group1, group2, use_method


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


def mannwhitney_test(group1: np.ndarray, group2: np.ndarray, 
                     large_sample_config: Dict[str, Any] = None) -> Tuple[float, str]:
    """
    Perform Mann-Whitney U test (non-parametric) between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        large_sample_config: Configuration for large sample handling
        
    Returns:
        Tuple of (p-value, test_name)
    """
    if large_sample_config is None:
        large_sample_config = {}
    
    try:
        # Handle large samples
        proc_group1, proc_group2, method = _handle_large_samples(
            group1, group2, large_sample_config
        )
        
        suppress = large_sample_config.get('suppress_warnings', False)
        with suppress_warnings_context(suppress):
            statistic, p_value = stats.mannwhitneyu(
                proc_group1, proc_group2, 
                alternative='two-sided',
                method=method
            )
        
        method_note = f" ({method})" if method != 'auto' else ""
        return p_value, f"Mann-Whitney U{method_note}"
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


def kruskal_test(groups_data: Dict[str, np.ndarray], 
                 large_sample_config: Dict[str, Any] = None) -> Tuple[float, str]:
    """
    Perform Kruskal-Wallis test (non-parametric) for multiple groups.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        large_sample_config: Configuration for large sample handling
        
    Returns:
        Tuple of (p-value, test_name)
    """
    if large_sample_config is None:
        large_sample_config = {}
    
    try:
        groups_list = [data for data in groups_data.values() if len(data) > 0]
        if len(groups_list) < 2:
            return np.nan, "Kruskal-Wallis"
        
        # Handle large samples by sampling down if needed
        max_exact = large_sample_config.get('max_exact_sample_size', 5000)
        sample_down = large_sample_config.get('sample_down_for_exact', False)
        
        if sample_down:
            processed_groups = []
            for data in groups_list:
                if len(data) > max_exact:
                    np.random.seed(42)
                    processed_groups.append(np.random.choice(data, size=max_exact, replace=False))
                    tqdm.write(f"  ⚠ Sampled down to {max_exact} for Kruskal-Wallis")
                else:
                    processed_groups.append(data)
            groups_list = processed_groups
        
        suppress = large_sample_config.get('suppress_warnings', False)
        with suppress_warnings_context(suppress):
            statistic, p_value = stats.kruskal(*groups_list)
        
        return p_value, "Kruskal-Wallis"
    except Exception as e:
        tqdm.write(f"  ⚠ Error in Kruskal-Wallis: {e}")
        return np.nan, "Kruskal-Wallis"


def get_pairwise_comparisons(groups_data: Dict[str, np.ndarray], 
                             test_func, apply_correction: bool = True,
                             large_sample_config: Dict[str, Any] = None) -> Dict[Tuple[str, str], Tuple[float, str]]:
    """
    Get pairwise comparisons between all groups.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        test_func: Function to use for pairwise tests (t_test or mannwhitney_test)
        apply_correction: Whether to apply Bonferroni correction
        large_sample_config: Configuration for large sample handling
        
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
            
            # Pass large_sample_config if test_func accepts it
            if large_sample_config and hasattr(test_func, '__code__'):
                # Check if function accepts large_sample_config parameter
                import inspect
                sig = inspect.signature(test_func)
                if 'large_sample_config' in sig.parameters:
                    p_value, test_name = test_func(data1, data2, large_sample_config=large_sample_config)
                else:
                    p_value, test_name = test_func(data1, data2)
            else:
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
                              test_config: Dict[str, Any],
                              excluded_groups: List[str] = None) -> Dict[str, Any]:
    """
    Main function to perform statistical tests between groups.
    Auto-selects appropriate test based on group count and data distribution.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        test_config: Statistical test configuration
        excluded_groups: List of group names to exclude from testing
        
    Returns:
        Dictionary with test results including p-values, test names, and pairwise comparisons
    """
    if not test_config.get('enabled', False):
        return {}
    
    # Filter out excluded groups
    if excluded_groups:
        groups_data = {k: v for k, v in groups_data.items() if k not in excluded_groups}
    
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
    
    # Get large sample configuration
    large_sample_config = {
        'large_sample_threshold': test_config.get('large_sample_threshold', 5000),
        'method': test_config.get('method', 'auto'),
        'max_exact_sample_size': test_config.get('max_exact_sample_size', 5000),
        'sample_down_for_exact': test_config.get('sample_down_for_exact', False),
        'suppress_warnings': test_config.get('suppress_warnings', False)
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
                p_value, test_name = mannwhitney_test(
                    list(groups_data.values())[0], 
                    list(groups_data.values())[1],
                    large_sample_config=large_sample_config
                )
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
                p_value, test_name = kruskal_test(groups_data, large_sample_config=large_sample_config)
                results["overall_test"] = {"p_value": p_value, "test": test_name}
                
                # Pairwise comparisons with Mann-Whitney
                pairwise = get_pairwise_comparisons(
                    groups_data, mannwhitney_test, 
                    apply_correction=True,
                    large_sample_config=large_sample_config
                )
                results["pairwise_comparisons"] = pairwise
    else:
        # Manual test selection (not implemented in this version, use auto_select)
        tqdm.write("  ⚠ Manual test selection not yet implemented, using auto-select")
    
    return results


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    d = (mean1 - mean2) / pooled_std
    return d


def eta_squared(groups_data: Dict[str, np.ndarray]) -> float:
    """
    Calculate eta-squared effect size for ANOVA.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        
    Returns:
        Eta-squared value
    """
    groups_list = [data for data in groups_data.values() if len(data) > 0]
    if len(groups_list) < 2:
        return np.nan
    
    # Calculate overall mean
    all_data = np.concatenate(groups_list)
    grand_mean = np.mean(all_data)
    
    # Calculate SS_total
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    if ss_total == 0:
        return np.nan
    
    # Calculate SS_between
    ss_between = 0
    for group_data in groups_list:
        n = len(group_data)
        group_mean = np.mean(group_data)
        ss_between += n * (group_mean - grand_mean) ** 2
    
    eta_sq = ss_between / ss_total
    return eta_sq


def epsilon_squared(groups_data: Dict[str, np.ndarray], h_statistic: float) -> float:
    """
    Calculate epsilon-squared effect size for Kruskal-Wallis test.
    
    Args:
        groups_data: Dictionary mapping group names to data arrays
        h_statistic: H statistic from Kruskal-Wallis test
        
    Returns:
        Epsilon-squared value
    """
    groups_list = [data for data in groups_data.values() if len(data) > 0]
    if len(groups_list) < 2:
        return np.nan
    
    n_total = sum(len(g) for g in groups_list)
    
    if n_total == 0:
        return np.nan
    
    epsilon_sq = (h_statistic - len(groups_list) + 1) / (n_total - len(groups_list))
    
    # Ensure non-negative
    return max(0, epsilon_sq)

