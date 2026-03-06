#!/usr/bin/env python3
"""
bench-stat: Benchmark Analysis & Comparison Tool

Analyzes benchmark results using statistical tests and text-based visualizations.

Usage:
    Single experiment analysis:
        bench-stat results.csv

    Compare two experiments:
        bench-stat baseline.csv experiment.csv

Options:
    --lower-is-better    Lower values = better performance
    --alpha FLOAT        Significance level (default: 0.05)
    --base-label TEXT    Label for baseline
    --exp-label TEXT     Label for experiment

Examples:
    bench-stat crypto.aes.base.csv
    bench-stat latency_base.csv latency_fix.csv --lower-is-better --alpha 0.01
"""

import argparse
import math
import sys
import warnings
import numpy as np
from scipy import stats as sp_stats
from scipy import optimize as sp_optimize

__all__ = [
    # Data loading
    'load_csv',
    # Descriptive statistics
    'compute_descriptive_stats',
    # Statistical tests
    'welch_t_test',
    'mann_whitney_u',
    'permutation_test',
    'bootstrap_ci',
    'bootstrap_single_ci',
    'dagostino_pearson_test',
    # Effect size and power
    'statistical_power',
    'minimum_detectable_effect',
    'required_sample_size',
    # Outlier detection
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    # Report generation
    'generate_single_report',
    'generate_report',
    # Interpretation helpers
    'significance_marker',
    'interpret_cohens_d',
    'one_tailed_p_value',
    'cumulative_comparison',
]


# =============================================================================
# Data Loading
# =============================================================================

def load_csv(filename):
    """Load single-column numeric CSV file."""
    raw = np.atleast_1d(np.genfromtxt(filename, invalid_raise=False))
    nan_count = int(np.sum(np.isnan(raw)))
    data = raw[~np.isnan(raw)]
    if nan_count > 0:
        warnings.warn(
            "{}: {} non-numeric value(s) skipped".format(filename, nan_count),
            stacklevel=2)
    return data.tolist()


# =============================================================================
# Basic Statistics
# =============================================================================

def _pooled_stdev_from_params(n1, v1, n2, v2):
    """Pooled standard deviation from pre-computed sample sizes and variances."""
    if n1 + n2 < 3:
        return 0.0
    return math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))


# =============================================================================
# Consolidated Descriptive Statistics
# =============================================================================

def compute_descriptive_stats(data):
    """Compute all descriptive statistics using library functions for robustness.

    Returns a dict with: n, mean, var, stdev, se, median, min, max, range,
    p25, p75, iqr, skewness, ex_kurtosis.

    Requires at least 3 data points. Uses scipy.stats.describe for optimized
    single-pass computation of nobs, min, max, mean, and variance.
    """
    arr = np.asarray(data)
    if arr.size < 3:
        raise ValueError("data must contain at least 3 values")

    desc = sp_stats.describe(arr)

    n = desc.nobs
    m = float(desc.mean)
    var_ = float(desc.variance)  # ddof=1 by default
    sd = math.sqrt(var_)
    se = sd / math.sqrt(n)
    data_min, data_max = float(desc.minmax[0]), float(desc.minmax[1])

    med = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))

    # Derive unbiased skewness from desc (biased); always valid for n >= 3
    biased_sk = float(desc.skewness)
    sk = biased_sk * math.sqrt(n * (n - 1)) / (n - 2) if not np.isnan(biased_sk) else 0.0

    # Derive unbiased kurtosis from desc (biased); requires n >= 4
    ek = 0.0
    if n >= 4:
        biased_ek = float(desc.kurtosis)
        if not np.isnan(biased_ek):
            ek = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * biased_ek + 6)

    return {
        'n': n,
        'mean': m,
        'var': var_,
        'stdev': sd,
        'se': se,
        'median': med,
        'min': data_min,
        'max': data_max,
        'range': data_max - data_min,
        'p25': p25,
        'p75': p75,
        'iqr': p75 - p25,
        'skewness': sk,
        'ex_kurtosis': ek,
    }


# =============================================================================
# Confidence Interval Helpers
# =============================================================================

def _margin_of_error_from_se(se, alpha, df):
    """Compute margin of error from standard error and degrees of freedom."""
    t_crit_val = _t_critical(alpha, df)
    return t_crit_val * se


# =============================================================================
# Formatting Helpers for Descriptive Statistics
# =============================================================================

def _format_stats_single(stats, alpha):
    """Format descriptive statistics table for a single experiment.

    Returns (lines, ci_lo, ci_hi, ci_str) where ci_str is the pre-formatted
    CI string, reusable by the caller to avoid redundant formatting.
    """
    n = stats['n']
    m = stats['mean']
    se = stats['se']
    moe = _margin_of_error_from_se(se, alpha, n - 1)
    ci_lo, ci_hi = m - moe, m + moe
    ci_str = "[{:.2f}, {:.2f}]".format(ci_lo, ci_hi)

    rows = [
        ("n", str(n)),
        ("Mean", "{:.2f}".format(m)),
        ("Std Dev", "{:.2f}".format(stats['stdev'])),
        ("Std Error", "{:.2f}".format(se)),
        ("Median", "{:.2f}".format(stats['median'])),
        ("Min", "{:.2f}".format(stats['min'])),
        ("Max", "{:.2f}".format(stats['max'])),
        ("Range", "{:.2f}".format(stats['range'])),
        ("25th pct (Q1)", "{:.2f}".format(stats['p25'])),
        ("75th pct (Q3)", "{:.2f}".format(stats['p75'])),
        ("IQR", "{:.2f}".format(stats['iqr'])),
        ("Skewness", "{:.3f}".format(stats['skewness'])),
        ("Ex. Kurtosis", "{:.3f}".format(stats['ex_kurtosis'])),
        ("{:.0f}% CI (mean)".format(100 * (1 - alpha)),
         ci_str),
    ]

    lines = []
    lines.append("  {:<20}  {:>20}".format("Metric", "Value"))
    lines.append("  {}  {}".format(
        "\u2500" * 20, "\u2500" * 20))
    for name, val in rows:
        lines.append("  {:<20}  {:>20}".format(name, val))

    return lines, ci_lo, ci_hi, ci_str


def _format_stats_comparison(stats_base, stats_exp, label_base, label_exp):
    """Format descriptive statistics table for a comparison report."""
    row_specs = [
        ("n", str(stats_base['n']), str(stats_exp['n'])),
        ("Mean", "{:.2f}".format(stats_base['mean']),
         "{:.2f}".format(stats_exp['mean'])),
        ("Median", "{:.2f}".format(stats_base['median']),
         "{:.2f}".format(stats_exp['median'])),
        ("Std Dev", "{:.2f}".format(stats_base['stdev']),
         "{:.2f}".format(stats_exp['stdev'])),
        ("Min", "{:.2f}".format(stats_base['min']),
         "{:.2f}".format(stats_exp['min'])),
        ("Max", "{:.2f}".format(stats_base['max']),
         "{:.2f}".format(stats_exp['max'])),
        ("25th pct", "{:.2f}".format(stats_base['p25']),
         "{:.2f}".format(stats_exp['p25'])),
        ("75th pct", "{:.2f}".format(stats_base['p75']),
         "{:.2f}".format(stats_exp['p75'])),
        ("Skewness", "{:.3f}".format(stats_base['skewness']),
         "{:.3f}".format(stats_exp['skewness'])),
        ("Ex. Kurtosis", "{:.3f}".format(stats_base['ex_kurtosis']),
         "{:.3f}".format(stats_exp['ex_kurtosis'])),
    ]

    lines = []
    lines.append("  {:<15}  {:>15}  {:>15}".format(
        "Metric", label_base, label_exp))
    lines.append("  {}  {}  {}".format(
        "\u2500" * 15, "\u2500" * 15, "\u2500" * 15))
    for name, v1, v2 in row_specs:
        lines.append("  {:<15}  {:>15}  {:>15}".format(name, v1, v2))

    return lines


# =============================================================================
# Distribution Shape Interpretation
# =============================================================================

def interpret_cv(cv):
    """Return interpretation line for coefficient of variation."""
    if cv < 1:
        return "    \u2192 Very low variability \u2014 highly consistent results"
    elif cv < 3:
        return "    \u2192 Low variability \u2014 reasonably consistent results"
    elif cv < 5:
        return "    \u2192 Moderate variability"
    else:
        return ("    \u2192 High variability \u2014 results are noisy, "
                "consider more runs")


def _interpret_skewness(sk):
    """Return interpretation lines for skewness."""
    lines = []
    lines.append("  Skewness:  {:+.3f}".format(sk))
    if abs(sk) < 0.5:
        lines.append("    \u2192 Approximately symmetric")
    elif sk > 0:
        lines.append(
            "    \u2192 Right-skewed (tail extends to higher values)")
    else:
        lines.append(
            "    \u2192 Left-skewed (tail extends to lower values)")
    return lines


def _interpret_kurtosis(ek):
    """Return interpretation lines for excess kurtosis."""
    lines = []
    lines.append("  Kurtosis:  {:+.3f}".format(ek))
    if abs(ek) < 0.5:
        lines.append(
            "    \u2192 Approximately mesokurtic (normal-like tails)")
    elif ek > 0:
        lines.append(
            "    \u2192 Leptokurtic (heavier tails than normal, sharper peak)")
    else:
        lines.append(
            "    \u2192 Platykurtic (lighter tails than normal, flatter peak)")
    return lines


# =============================================================================
# Normality Assessment
# =============================================================================

def _normality_assessment(datasets, labels, edges, alpha, stats_list=None):
    """Generate normality assessment section for one or more datasets.

    Returns (lines, p_values) where p_values is a list of p-values per dataset.

    If stats_list is provided, pre-computed mean and stdev are reused by the
    histogram to avoid recalculation.
    """
    lines = []
    p_values = []

    for i, (data, label) in enumerate(zip(datasets, labels)):
        stats = stats_list[i] if stats_list else None
        lines.append(normality_histogram(data, label, edges, max_bar_width=20, stats=stats))

    # D'Agostino-Pearson test results
    results = []
    for data, label in zip(datasets, labels):
        k2, p_norm = dagostino_pearson_test(data)
        results.append((label, k2, p_norm))
        p_values.append(p_norm)

    if len(datasets) == 1:
        label, k2, p_norm = results[0]
        lines.append("  D'Agostino-Pearson Omnibus Test:")
        lines.append("    Statistic: {:.3f}".format(k2))
        lines.append("    p-value:   {:.4f}".format(p_norm))
        if p_norm > alpha:
            lines.append(
                "    Result:    "
                "\u2705 Consistent with normal distribution")
        else:
            lines.append(
                "    Result:    "
                "\u274c Significantly non-normal")
        lines.append("")
    else:
        lines.append("  D'Agostino-Pearson Omnibus Test:")
        lines.append(
            "  {:<20}  {:>10}  {:>10}  {:>15}".format(
                "Dataset", "Statistic", "p-value", "Normal?"))
        lines.append(
            "  {}  {}  {}  {}".format(
                "\u2500" * 20, "\u2500" * 10,
                "\u2500" * 10, "\u2500" * 15))
        for label, k2, p_norm in results:
            normal_str = ("\u2705 Yes" if p_norm > alpha
                          else "\u274c No")
            lines.append(
                "  {:<20}  {:>10.3f}  {:>10.4f}  {:>15}".format(
                    label, k2, p_norm, normal_str))
        lines.append("")

    return lines, p_values


# =============================================================================
# Outlier Detection (shared)
# =============================================================================


def format_outlier_report(stats, outliers_iqr, outliers_zscore):
    """Format outlier detection results."""
    p25 = stats['p25']
    p75 = stats['p75']
    iqr = stats['iqr']

    lines = []
    lines.append("  IQR Method (1.5\u00d7IQR rule):")
    lines.append("    Lower fence: {:.2f}".format(p25 - 1.5 * iqr))
    lines.append("    Upper fence: {:.2f}".format(p75 + 1.5 * iqr))
    if outliers_iqr:
        lines.append(
            "    Outliers ({}): {}".format(
                len(outliers_iqr),
                ', '.join('{:.2f}'.format(v) for v in outliers_iqr)))
    else:
        lines.append("    No outliers detected")
    lines.append("")

    lines.append("  Z-score Method (|z| > 3):")
    if outliers_zscore:
        for val, z in outliers_zscore:
            lines.append("    {:.2f} (z = {:+.2f})".format(val, z))
    else:
        lines.append("    No outliers detected")
    lines.append("")

    return lines


# =============================================================================
# Sample Size Analysis (shared core)
# =============================================================================

def _required_n_for_moe(s, target_moe, alpha):
    """Required sample size to achieve a target absolute margin of error.

    Uses the t-distribution with iterative refinement: start with z-based
    estimate, then adjust using t critical value at the estimated df.
    """
    if target_moe <= 0:
        return float('inf')
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    n_est = max(2, math.ceil((z * s / target_moe) ** 2))
    # Refine with t-distribution (2 iterations is sufficient for convergence)
    for _ in range(2):
        t_crit = float(abs(sp_stats.t.ppf(alpha / 2, n_est - 1)))
        n_est = max(2, math.ceil((t_crit * s / target_moe) ** 2))
    return n_est


def _required_n_for_relative_precision(m, s, alpha, target_pct=1.0):
    """Required sample size for target relative margin of error (percentage of mean)."""
    target_moe = abs(m) * (target_pct / 100.0)
    return _required_n_for_moe(s, target_moe, alpha)


def _precision_table(m, s, n, alpha):
    """Generate precision-vs-required-n table rows used by sample size analysis."""
    lines = []
    lines.append(
        "    {:>15}  {:>12}  {:>12}  {:>15}".format(
            "Target MoE %", "Target MoE", "Required n", "Status"))
    lines.append(
        "    {}  {}  {}  {}".format(
            "\u2500" * 15, "\u2500" * 12,
            "\u2500" * 12, "\u2500" * 15))

    targets_pct = [0.25, 0.5, 1.0, 2.0, 3.0]
    for target_pct in targets_pct:
        target_moe = abs(m) * target_pct / 100.0
        required_n = _required_n_for_moe(s, target_moe, alpha)

        if required_n == float('inf'):
            req_str = "\u221e"
            status = "\u274c N/A (mean=0)"
        elif n >= required_n:
            req_str = str(required_n)
            status = "\u2705 Achieved"
        elif n >= required_n * 0.7:
            req_str = str(required_n)
            status = "\u26a0\ufe0f  Close"
        else:
            req_str = str(required_n)
            status = "\u274c Need {} more".format(required_n - n)

        lines.append(
            "    {:>15}  {:>12}  {:>12}  {:>15}".format(
                "\u00b1{:.2f}%".format(target_pct),
                "\u00b1{:.2f}".format(target_moe),
                req_str,
                status))

    return lines


def _power_table(n1, n2, s_pool, observed_d, alpha):
    """Generate power analysis table rows used by comparison sample size analysis."""
    cohens_d_label = "Cohen's d"
    lines = []
    lines.append(
        "    {:>15}  {:>10}  {:>12}  {:>8}  {:>12}".format(
            "Effect Size", cohens_d_label, "Absolute",
            "Power", "Adequate?"))
    lines.append(
        "    {}  {}  {}  {}  {}".format(
            "\u2500" * 15, "\u2500" * 10,
            "\u2500" * 12, "\u2500" * 8, "\u2500" * 12))

    for d_label, d_val in _effect_size_entries(
            observed_d, min_observed=-1.0):
        abs_val = d_val * s_pool
        power = statistical_power(n1, n2, d_val, alpha)
        adequate = _threshold_label(power, 0.8, 0.5,
                                    "\u2705 Yes", "\u26a0\ufe0f  Marginal",
                                    "\u274c No")
        lines.append(
            "    {:>15}  {:>10.3f}  {:>12}  {:>7.1%}  {:>12}".format(
                d_label, d_val,
                "\u00b1{:.2f}".format(abs_val),
                power, adequate))

    return lines


def _required_n_table(s_pool, observed_d, n1, n2, alpha):
    """Generate required-n table rows used by comparison sample size analysis."""
    cohens_d_label = "Cohen's d"
    lines = []
    lines.append(
        "    {:>15}  {:>10}  {:>12}  {:>12}  {:>15}".format(
            "Effect Size", cohens_d_label, "Absolute",
            "Required n", "Status"))
    lines.append(
        "    {}  {}  {}  {}  {}".format(
            "\u2500" * 15, "\u2500" * 10,
            "\u2500" * 12, "\u2500" * 12, "\u2500" * 15))

    current_n = min(n1, n2)
    for d_label, d_val in _effect_size_entries(
            observed_d, min_observed=0.01):
        abs_val = d_val * s_pool
        req_n = required_sample_size(d_val, alpha)
        status = _required_n_status(current_n, req_n,
                                    "\u2705 Sufficient", "\u26a0\ufe0f  Close")
        lines.append(
            "    {:>15}  {:>10.3f}  {:>12}  {:>12}  {:>15}".format(
                d_label, d_val,
                "\u00b1{:.2f}".format(abs_val),
                req_n, status))

    return lines


def _effect_size_entries(observed_d, min_observed=0.0):
    """Return standard effect size rows used by comparison sample-size tables."""
    entries = [("Small", 0.2), ("Medium", 0.5), ("Large", 0.8)]
    if abs(observed_d) > min_observed:
        entries.append(("Observed", abs(observed_d)))
    return entries


def _threshold_label(value, high, medium, high_label, medium_label, low_label):
    """Map a numeric value into high/medium/low labels."""
    if value >= high:
        return high_label
    if value >= medium:
        return medium_label
    return low_label


def _required_n_status(current_n, required_n, sufficient_label, close_label):
    """Return adequacy status for current sample size vs required sample size."""
    if current_n >= required_n:
        return sufficient_label
    if current_n >= required_n * 0.7:
        return close_label
    return "\u274c Need {} more".format(required_n - current_n)


def _sample_size_analysis_single(stats, alpha=0.05, moe=None, ci_lo=None, ci_hi=None):
    """Analyze whether sample size is adequate for a single experiment.

    If moe/ci bounds are provided, they are reused to avoid recomputation.
    """
    n = stats['n']
    m = stats['mean']
    s = stats['stdev']
    se = stats['se']
    if moe is None:
        moe = _margin_of_error_from_se(se, alpha, n - 1)
    if m != 0:
        moe_pct = 100.0 * moe / abs(m)
    else:
        moe_pct = float('inf')
    if ci_lo is None or ci_hi is None:
        ci_lo, ci_hi = m - moe, m + moe

    lines = []
    lines.append("  Current sample size: n = {}".format(n))
    lines.append("  Mean:                {:.2f}".format(m))
    lines.append("  Std Dev:             {:.2f}".format(s))
    lines.append("  Std Error:           {:.2f}".format(se))
    if m != 0:
        lines.append("  Margin of Error:     \u00b1{:.2f} (\u00b1{:.2f}% of mean)".format(
            moe, moe_pct))
    else:
        lines.append(
            "  Margin of Error:     \u00b1{:.2f} (mean is zero, relative % N/A)".format(
                moe))
    lines.append("")

    lines.append("  Precision of mean estimate:")
    lines.append("    True mean is within \u00b1{:.2f} of {:.2f}".format(moe, m))
    lines.append("    i.e., between {:.2f} and {:.2f}".format(ci_lo, ci_hi))
    lines.append("    with {:.0f}% confidence".format(100 * (1 - alpha)))
    lines.append("")

    if m != 0:
        lines.append("  Samples needed for target precision:")
        lines.extend(_precision_table(m, s, n, alpha))
        lines.append("")

    lines.append("  Assessment:")
    if m != 0:
        if moe_pct <= 0.5:
            lines.append(
                "  \u2705 Excellent precision \u2014 mean is estimated very accurately")
        elif moe_pct <= 1.0:
            lines.append(
                "  \u2705 Good precision \u2014 sufficient for most comparisons")
        elif moe_pct <= 2.0:
            lines.append(
                "  \u26a0\ufe0f  Moderate precision \u2014 "
                "may miss small differences in comparisons")
        else:
            req_1pct = _required_n_for_relative_precision(m, s, alpha, target_pct=1.0)
            if moe_pct <= 5.0:
                lines.append(
                    "  \u26a0\ufe0f  Low precision \u2014 consider collecting more samples")
            else:
                lines.append(
                    "  \u274c Poor precision \u2014 results are unreliable, "
                    "collect more data")
            lines.append(
                "      To reach \u00b11% precision, you need ~{} samples".format(req_1pct))
    else:
        lines.append(
            "  \u26a0\ufe0f  Mean is zero \u2014 relative precision is not applicable")
        lines.append(
            "  Absolute margin of error: \u00b1{:.2f}".format(moe))
    lines.append("")

    return '\n'.join(lines)


def _sample_size_analysis_comparison(stats_base, stats_exp,
                                    label_base, label_exp, alpha=0.05,
                                    s_pool=None, observed_d=None):
    """Analyze whether sample sizes are adequate for detecting differences.

    If s_pool and/or observed_d are provided, they are reused to avoid
    recomputing pooled standard deviation and effect size.
    """
    n1 = stats_base['n']
    n2 = stats_exp['n']
    m1 = stats_base['mean']

    if s_pool is None:
        s_pool = _pooled_stdev_from_params(
            n1, stats_base['var'], n2, stats_exp['var'])

    if observed_d is None:
        mean_diff = stats_exp['mean'] - stats_base['mean']
        observed_d = (mean_diff / s_pool if s_pool != 0.0 else 0.0)

    lines = []
    lines.append("  Sample sizes:  {}: n={},  {}: n={}".format(
        label_base, n1, label_exp, n2))
    lines.append("  Pooled Std Dev: {:.2f}".format(s_pool))
    lines.append("")

    min_d = minimum_detectable_effect(n1, n2, alpha)
    min_abs = min_d * s_pool

    lines.append("  Minimum Detectable Effect (at 80% power):")
    lines.append("    Cohen's d:  {:.3f} ({})".format(
        min_d, interpret_cohens_d(min_d)))
    lines.append("    Absolute:   \u00b1{:.2f} ops/min".format(min_abs))
    if m1 != 0:
        lines.append("    Relative:   \u00b1{:.2f}% of baseline mean".format(
            100 * min_abs / abs(m1)))
    else:
        lines.append("    Relative:   N/A (baseline mean is zero)")
    lines.append("")

    obs_power = (statistical_power(n1, n2, abs(observed_d), alpha)
                 if abs(observed_d) > 0 else 0.0)

    if abs(observed_d) > 0:
        lines.append("  Power for observed effect (d={:.3f}):".format(
            observed_d))
        lines.append("    Power: {:.1%}".format(obs_power))
        if obs_power >= 0.8:
            lines.append(
                "    \u2705 Adequate power to detect this effect")
        elif obs_power >= 0.5:
            lines.append(
                "    \u26a0\ufe0f  Moderate power \u2014 "
                "result is suggestive but could be missed")
        else:
            lines.append(
                "    \u274c Low power \u2014 "
                "this effect size is hard to detect with current n")
        lines.append("")

    lines.append("  Power analysis for standard effect sizes:")
    lines.extend(_power_table(n1, n2, s_pool, observed_d, alpha))
    lines.append("")

    lines.append("  Required n (per group) for 80% power:")
    lines.extend(_required_n_table(s_pool, observed_d, n1, n2, alpha))
    lines.append("")

    lines.append("  Assessment:")

    if min(n1, n2) < 10:
        lines.append(
            "  \u274c INSUFFICIENT: Very small sample size (n < 10)")
        lines.append(
            "     Results are unreliable. "
            "Collect at least 30 samples per group.")
    elif min(n1, n2) < 30:
        lines.append("  \u26a0\ufe0f  SMALL SAMPLE: n < 30")
        lines.append(
            "     Can detect large effects, "
            "but may miss small improvements.")
        lines.append(
            "     Can reliably detect: d \u2265 {:.2f} "
            "({:.2f} ops/min)".format(min_d, min_abs))
    elif obs_power >= 0.8:
        lines.append(
            "  \u2705 ADEQUATE: Sample size is sufficient "
            "for the observed effect.")
    else:
        lines.append(
            "  \u26a0\ufe0f  UNDERPOWERED for small effects.")
        lines.append(
            "     Can reliably detect: d \u2265 {:.2f} "
            "({:.2f} ops/min)".format(min_d, min_abs))
        req = required_sample_size(0.2, alpha)
        lines.append(
            "     For small effects (d=0.2), "
            "need n={} per group".format(req))

    if max(n1, n2) > 1.5 * min(n1, n2):
        lines.append("")
        lines.append(
            "  \u26a0\ufe0f  Unequal sample sizes (n={} vs n={})".format(
                n1, n2))
        lines.append(
            "     Consider collecting equal-sized samples "
            "for maximum power.")

    lines.append("")
    return '\n'.join(lines)


# =============================================================================
# Probability Distributions
# =============================================================================

def _t_critical(alpha, df):
    """t critical value for two-tailed CI using scipy.stats.t.ppf."""
    return float(abs(sp_stats.t.ppf(alpha / 2, df)))


# =============================================================================
# Statistical Tests
# =============================================================================

def welch_t_test(data1, data2):
    """Welch's two-sample t-test using scipy.stats.ttest_ind.

    Returns (t_stat, df, p_two_tailed).
    Tests whether data2 is significantly different from data1.
    t_stat > 0 means data2 > data1 on average.
    """
    arr1 = np.asarray(data1)
    arr2 = np.asarray(data2)

    # scipy computes data1 - data2; we want data2 - data1, so negate t_stat
    result = sp_stats.ttest_ind(arr1, arr2, equal_var=False)
    t_stat = -float(result.statistic)
    df = float(result.df)
    p_two = float(result.pvalue)

    # Handle NaN from zero-variance edge cases (both groups constant)
    if math.isnan(t_stat) or math.isnan(df):
        mean_diff = float(arr2.mean()) - float(arr1.mean())
        df = float(len(data1) + len(data2) - 2)
        if mean_diff != 0:
            t_stat = float('inf') if mean_diff > 0 else float('-inf')
            p_two = 0.0
        else:
            t_stat = 0.0
            p_two = 1.0

    return t_stat, df, p_two


def mann_whitney_u(data1, data2):
    """Mann-Whitney U test using scipy.stats.mannwhitneyu with tie correction.

    Returns (U, z_stat, p_two_tailed).

    The z-statistic uses tie-corrected variance and continuity correction to
    ensure consistency with scipy's p-value calculation.
    """
    # Convert once and reuse to avoid duplicate array conversions.
    arr1 = np.asarray(data1)
    arr2 = np.asarray(data2)
    n1, n2 = len(arr1), len(arr2)

    # scipy returns U for data1 (first argument)
    u1, p_value = sp_stats.mannwhitneyu(arr1, arr2, alternative='two-sided')

    # Compute z-statistic for data2 direction with tie correction and continuity correction
    mu = n1 * n2 / 2.0
    u2 = n1 * n2 - u1

    # Apply tie correction to variance
    combined = np.concatenate([arr1, arr2])
    n_total = len(combined)
    _, counts = np.unique(combined, return_counts=True)
    tie_sum = np.sum(counts * (counts - 1) * (counts + 1))

    var_basic = n1 * n2 * (n1 + n2 + 1) / 12.0
    correction = 1 - tie_sum / (n_total * (n_total - 1) * (n_total + 1)) if n_total > 1 else 1.0
    sigma = math.sqrt(var_basic * correction)

    if sigma == 0:
        return float(u1), 0.0, 1.0

    # Apply continuity correction: move toward mu by 0.5
    # The direction of continuity correction depends on whether u2 is below or above mu
    if u2 < mu:
        z = (u2 - mu + 0.5) / sigma
    else:
        z = (u2 - mu - 0.5) / sigma

    return float(u1), float(z), float(p_value)


def permutation_test(data1, data2, n_perms=10000, seed=42):
    """Permutation test for difference in means using scipy.stats.permutation_test.

    Returns (observed_diff, p_two_tailed).

    Uses scipy's native implementation which supports exact permutations for
    small samples and optimized randomized permutations for large samples.
    """
    arr1 = np.asarray(data1)
    arr2 = np.asarray(data2)
    observed_diff = float(arr2.mean() - arr1.mean())

    def statistic(x, y, axis):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    result = sp_stats.permutation_test(
        (arr1, arr2), statistic, n_resamples=n_perms,
        alternative='two-sided', vectorized=True,
        rng=np.random.default_rng(seed))

    return observed_diff, float(result.pvalue)


def bootstrap_ci(data1, data2, n_boot=10000, ci=95, seed=42):
    """Bootstrap confidence interval for difference in means.

    Returns (ci_lower, ci_upper, p_two_tailed).

    Uses scipy.stats.bootstrap with BCa (bias-corrected and accelerated) method
    for better coverage on skewed distributions. The p-value is computed from
    the bootstrap distribution of mean differences.
    Falls back to simple computation for constant data where BCa is undefined.
    """
    data1_arr = np.asarray(data1)
    data2_arr = np.asarray(data2)

    # BCa requires non-constant data in at least one group
    if np.ptp(data1_arr) == 0 and np.ptp(data2_arr) == 0:
        diff = float(data2_arr.mean() - data1_arr.mean())
        if diff == 0:
            return 0.0, 0.0, 1.0
        return diff, diff, 0.0

    confidence_level = ci / 100.0

    def mean_diff(x, y, axis):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    result = sp_stats.bootstrap(
        (data1_arr, data2_arr), mean_diff, n_resamples=n_boot,
        confidence_level=confidence_level, method='BCa', paired=False,
        vectorized=True, rng=np.random.default_rng(seed))

    ci_lower = float(result.confidence_interval.low)
    ci_upper = float(result.confidence_interval.high)

    # Compute p-value from bootstrap distribution
    diffs = result.bootstrap_distribution.ravel()
    n = len(diffs)
    p_left = np.sum(diffs < 0) / n
    p_right = np.sum(diffs > 0) / n
    p_equal = 1.0 - p_left - p_right
    p_two = float(min(1.0, 2.0 * (min(p_left, p_right) + p_equal / 2)))

    return ci_lower, ci_upper, p_two


def bootstrap_single_ci(data, n_boot=10000, ci=95, seed=42):
    """Bootstrap confidence interval for the mean of a single sample.

    Uses scipy.stats.bootstrap with BCa (bias-corrected and accelerated) method
    for better coverage on skewed distributions.
    Falls back for constant data where BCa is undefined.
    """
    data_arr = np.asarray(data)
    m = float(data_arr.mean())

    # BCa requires non-constant data
    if np.ptp(data_arr) == 0:
        return m, m

    confidence_level = ci / 100.0
    result = sp_stats.bootstrap(
        (data_arr,), np.mean, n_resamples=n_boot,
        confidence_level=confidence_level, method='BCa',
        rng=np.random.default_rng(seed))

    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def dagostino_pearson_test(data):
    """D'Agostino-Pearson omnibus normality test using scipy.stats.normaltest.

    Returns (statistic, p_value).
    """
    n = len(data)
    # scipy.stats.normaltest uses kurtosistest which requires n>=20
    if n < 20:
        return 0, 1.0
    # Constant data has no meaningful distribution to test
    if np.max(data) == np.min(data):
        return 0, 1.0
    k2, p_value = sp_stats.normaltest(data)
    if np.isnan(p_value):
        return 0, 1.0
    return float(k2), float(p_value)


# =============================================================================
# Power Analysis
# =============================================================================

def statistical_power(n1, n2, d, alpha=0.05):
    """Compute exact power for two-sample t-test using non-central t-distribution.

    Power is P(reject H0 | effect size is d). When d=0 (null hypothesis is true),
    power equals alpha (the Type I error rate): P(reject true H0).

    Args:
        n1, n2: Sample sizes
        d: Cohen's effect size (expected difference / pooled std dev)
        alpha: Significance level (default 0.05)

    Returns:
        Power estimate in [0, 1]
    """
    if d == 0:
        # Under null hypothesis, power = P(Type I error) = alpha
        return alpha
    n_eff = 2.0 / (1.0 / n1 + 1.0 / n2)
    ncp = d * math.sqrt(n_eff / 2.0)
    df = n1 + n2 - 2
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
    # P(reject H0) = P(|T| > t_crit) under non-central t
    power = 1.0 - sp_stats.nct.cdf(t_crit, df, ncp) + sp_stats.nct.cdf(-t_crit, df, ncp)
    return max(0.0, min(1.0, float(power)))


def minimum_detectable_effect(n1, n2, alpha=0.05, target_power=0.80):
    """Find the minimum Cohen's d detectable with given sample sizes."""
    # Handle trivial/degenerate targets and cap behavior consistently.
    if target_power <= alpha:
        return 0.0

    d_lo, d_hi = 0.0, 3.0
    f = lambda d: statistical_power(n1, n2, d, alpha) - target_power

    # If even very large effects do not hit target power, return the cap.
    if f(d_hi) < 0:
        return d_hi

    return float(sp_optimize.bisect(f, d_lo, d_hi, xtol=0.001, maxiter=100))


def required_sample_size(d, alpha=0.05, target_power=0.80):
    """Find required n (per group) to detect effect size d."""
    if d <= 0:
        return float('inf')

    n_lo, n_hi = 2.0, 10000.0
    f = lambda n: statistical_power(n, n, d, alpha) - target_power

    # If minimum configured sample size already meets power target.
    if f(n_lo) >= 0:
        return int(n_lo)

    # Mirror existing behavior if target is not reached within the configured cap.
    if f(n_hi) < 0:
        return int(n_hi)

    # Solve in continuous n, then round up to an integer sample size.
    n_cont = float(sp_optimize.brentq(f, n_lo, n_hi, maxiter=100))
    n_req = max(2, int(math.ceil(n_cont)))

    # Ensure rounding still satisfies target power.
    while statistical_power(n_req, n_req, d, alpha) < target_power and n_req < int(n_hi):
        n_req += 1

    return n_req


# =============================================================================
# Outlier Detection
# =============================================================================

def detect_outliers_iqr(data, q1, q3, iqr):
    """Detect outliers using IQR method (vectorized)."""
    arr = np.asarray(data)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (arr < lower) | (arr > upper)
    return sorted(arr[mask].tolist())


def detect_outliers_zscore(data, m, s, threshold=3.0):
    """Detect outliers using z-score method (vectorized)."""
    if s == 0:
        return []
    arr = np.asarray(data)
    z_scores = (arr - m) / s
    mask = np.abs(z_scores) > threshold
    outliers = [(float(arr[i]), float(z_scores[i])) for i in np.where(mask)[0]]
    outliers.sort(key=lambda x: abs(x[1]), reverse=True)
    return outliers


# =============================================================================
# Histogram Helpers
# =============================================================================

def _compute_bins(data_sets):
    """Compute bin edges using numpy's optimized histogram_bin_edges.

    Uses numpy's 'auto' method which selects the best of Sturges and
    Freedman-Diaconis algorithms, then enforces min/max bin count constraints
    for readability.

    Enforces min_bins=8 and max_bins=20 to ensure histograms are readable
    and not too sparse or cluttered.
    """
    all_data = np.concatenate([np.asarray(d) for d in data_sets])

    # Handle constant data (zero range)
    if np.ptp(all_data) == 0:
        return [float(all_data[0]) - 1, float(all_data[0]) + 1]

    # Use numpy's optimized bin edge computation
    edges = np.histogram_bin_edges(all_data, bins='auto')
    n_bins = len(edges) - 1

    # Enforce min/max bin count for readable histograms
    if n_bins < 8:
        edges = np.histogram_bin_edges(all_data, bins=8)
    elif n_bins > 20:
        edges = np.histogram_bin_edges(all_data, bins=20)

    return edges.tolist()


def _bin_data(data, edges):
    """Assign data to bins using numpy's histogram.

    Last bin is closed on both sides: [low, high]. numpy.histogram handles
    this correctly by default.
    """
    counts, _ = np.histogram(data, bins=edges)
    return counts.tolist()


def _expected_normal_counts(n, m, s, edges):
    """Expected counts per bin under a normal distribution."""
    if s <= 0:
        expected = [0.0] * (len(edges) - 1)
        for i in range(len(edges) - 1):
            if edges[i] <= m < edges[i + 1]:
                expected[i] = float(n)
                return expected
        if m == edges[-1] and expected:
            expected[-1] = float(n)
        return expected

    edges_arr = np.asarray(edges)
    cdf_vals = sp_stats.norm.cdf(edges_arr, loc=m, scale=s)
    expected = np.diff(cdf_vals) * n
    return expected.tolist()


def _format_bin_label(low, high):
    """Format a bin label like [1050,1055)."""
    for val in [low, high]:
        if val != int(val):
            return "[{:.1f},{:.1f})".format(low, high)
    return "[{},{})".format(int(low), int(high))


def _compute_bin_labels(edges):
    """Pre-compute all bin labels for given edges. Avoids redundant list recreation."""
    return [_format_bin_label(edges[i], edges[i + 1])
            for i in range(len(edges) - 1)]


# =============================================================================
# Text Histogram Generation
# =============================================================================

def _get_mean_stdev(data, stats=None):
    """Extract mean and stdev from stats dict, or compute from data.

    Returns: (m, s) tuple where m is mean and s is stdev (sample std, ddof=1)
    """
    m = stats['mean'] if stats else float(np.mean(data))
    s = stats['stdev'] if stats else float(np.std(data, ddof=1))
    return m, s


def _histogram_setup(data, edges, stats=None):
    """Extract common histogram setup logic.

    Returns: (counts, m, s, n, labels, label_width)
    """
    counts = _bin_data(data, edges)
    m, s = _get_mean_stdev(data, stats)
    n = stats['n'] if stats else len(data)
    labels = _compute_bin_labels(edges)
    label_width = max(len(l) for l in labels)
    return counts, m, s, n, labels, label_width


def _single_histogram(data, label, edges, bar_char='\u2588', max_bar_width=40, stats=None):
    """Generate a single text histogram.

    If stats is provided (from compute_descriptive_stats), pre-computed mean and
    stdev are reused to avoid recalculation.
    """
    counts, m, s, _, labels, label_width = _histogram_setup(data, edges, stats)
    max_count = max(counts) if counts else 1

    lines = []
    lines.append("  {}".format(label))
    lines.append("  {:<{w}}  {:>10}".format(
        "ops/min", "Frequency", w=label_width))

    for i, (lbl, cnt) in enumerate(zip(labels, counts)):
        bar_len = (int(cnt / max_count * max_bar_width)
                   if max_count > 0 else 0)
        bar = bar_char * bar_len

        markers = []
        if edges[i] <= m < edges[i + 1]:
            markers.append("mean ({:.2f})".format(m))
        if cnt == max_count and cnt > 0:
            markers.append("mode")

        marker_str = ("  \u2190 {}".format(', '.join(markers))
                      if markers else "")
        lines.append(
            "  {:<{w}} \u2502 {:<{bw}} {:>3}{}".format(
                lbl, bar, cnt, marker_str,
                w=label_width, bw=max_bar_width))

    lines.append("  {:<{w}} \u2514{}\u2500".format(
        "", "\u2500" * (max_bar_width + 4), w=label_width))
    lines.append(
        "  {:<{w}}   Mean = {:.2f}  Std = {:.2f}  n = {}".format(
            "", m, s, len(data), w=label_width))
    lines.append("")
    return '\n'.join(lines)


def normality_histogram(data, label, edges, max_bar_width=20, stats=None):
    """Generate histogram comparing actual vs expected normal distribution.

    If stats is provided (from compute_descriptive_stats), pre-computed mean and
    stdev are reused to avoid recalculation.
    """
    counts, m, s, n, labels, label_width = _histogram_setup(data, edges, stats)
    expected = _expected_normal_counts(n, m, s, edges)
    max_val = max(max(counts), max(expected)) if counts else 1

    lines = []
    lines.append(
        "  {} \u2014 Actual vs. Normal Expected".format(label))
    lines.append(
        "  {:<{w}}  {:>6}  {:>8}  Shape comparison".format(
            "ops/min", "Actual", "Expected", w=label_width))

    for i, lbl in enumerate(labels):
        act = counts[i]
        exp = expected[i]
        act_bar = (int(act / max_val * max_bar_width)
                   if max_val > 0 else 0)
        exp_bar = (int(exp / max_val * max_bar_width)
                   if max_val > 0 else 0)

        act_str = '\u2588' * act_bar
        exp_str = '\u2591' * exp_bar

        if act > exp * 1.3 and act - exp > 1:
            note = "\u2190 EXCESS"
        elif act < exp * 0.7 and exp - act > 1:
            note = "\u2190 DEFICIT"
        else:
            note = "\u2248 match"

        lines.append(
            "  {:<{w}} \u2502 {:<{bw}} {:>5.0f}  \u2502 "
            "{:<{bw}} {:>5.1f}   {}".format(
                lbl, act_str, act, exp_str, exp, note,
                w=label_width, bw=max_bar_width))

    lines.append("  {:<{w}} \u2514{}\u2500".format(
        "", "\u2500" * (max_bar_width + 6), w=label_width))
    lines.append(
        "  {:<{w}}   \u2588 = Actual    \u2591 = Expected Normal".format(
            "", w=label_width))
    lines.append("")
    return '\n'.join(lines)


def overlay_histogram(data1, data2, label1, label2, edges,
                      max_bar_width=50):
    """Generate side-by-side overlay histogram."""
    counts1 = _bin_data(data1, edges)
    counts2 = _bin_data(data2, edges)

    labels = _compute_bin_labels(edges)
    label_width = max(len(l) for l in labels)

    c1 = label1[0].upper()
    c2 = label2[0].upper()
    if c1 == c2:
        c2 = label2[0].lower()

    lines = []
    lines.append("  Side-by-Side: {} vs {}".format(label1, label2))
    lines.append("  {}={}  {}={}  \u2592=Overlap".format(
        c1, label1, c2, label2))
    lines.append("")

    for i in range(len(labels)):
        cnt1, cnt2 = counts1[i], counts2[i]
        overlap = min(cnt1, cnt2)
        only1 = cnt1 - overlap
        only2 = cnt2 - overlap
        bar = '\u2592' * overlap + c1 * only1 + c2 * only2

        lines.append(
            "  {:<{w}} \u2502 {:<{bw}} {}:{:>2}  {}:{:>2}".format(
                labels[i], bar, c1, cnt1, c2, cnt2,
                w=label_width, bw=max_bar_width))

    lines.append("  {:<{w}} \u2514{}".format(
        "", "\u2500" * (max_bar_width + 12), w=label_width))
    lines.append("")
    return '\n'.join(lines)


def shape_ascii_art(data, edges, width=20, height=5):
    """Generate a small ASCII art shape of the distribution."""
    counts = _bin_data(data, edges)
    if not counts:
        return [""] * height

    max_c = max(counts)
    if max_c == 0:
        return [" " * width] * height

    normalized = [c / max_c * height for c in counts]
    cols = width
    bin_count = len(counts)
    col_values = []
    for col in range(cols):
        bin_idx = int(col * bin_count / cols)
        bin_idx = min(bin_idx, bin_count - 1)
        col_values.append(normalized[bin_idx])

    lines = []
    for row in range(height, 0, -1):
        line = ""
        for col in range(cols):
            if col_values[col] >= row:
                if col_values[col] >= row + 0.5:
                    line += '\u2588'
                else:
                    line += '\u2584'
            elif col_values[col] >= row - 0.5:
                line += '\u2584'
            else:
                line += ' '
        lines.append(line)
    return lines


def _distribution_shape_comparison(datasets, labels, edges,
                                  art_width=20, art_height=5):
    """Generate side-by-side ASCII art distribution shapes."""
    arts = []
    for data in datasets:
        arts.append(shape_ascii_art(data, edges, art_width, art_height))

    lines = []
    header = ""
    for label in labels:
        header += "  {:<{w}}".format(label, w=art_width + 2)
    lines.append(header)
    lines.append("")

    for row in range(art_height):
        line = ""
        for art in arts:
            if row < len(art):
                line += "  {:<{w}}".format(art[row], w=art_width + 2)
            else:
                line += "  {:<{w}}".format("", w=art_width + 2)
        lines.append(line)

    return '\n'.join(lines)


# =============================================================================
# Stability Chart
# =============================================================================

def _stability_chart(data, label, width=60, stats=None):
    """Generate a run chart showing values over time with control limits.

    If stats is provided (from compute_descriptive_stats), pre-computed mean and
    stdev are reused to avoid recalculation.
    """
    if not data:
        return "  No data available for stability chart\n"
    n = stats['n'] if stats else len(data)
    m, s = _get_mean_stdev(data, stats)
    upper = m + 2 * s
    lower = m - 2 * s
    data_min = min(stats['min'] if stats else min(data), lower)
    data_max = max(stats['max'] if stats else max(data), upper)
    data_range = data_max - data_min

    if data_range == 0:
        return "  All values identical \u2014 perfectly stable\n"

    lines = []
    lines.append(
        "  {} \u2014 Run Chart (chronological order)".format(label))
    lines.append("  Each row = one measurement, position shows value")
    lines.append("")

    lines.append(
        "  {:>4}  {:>8}  {:<8.1f}{:^{cw}}{:>8.1f}".format(
            "#", "Value", data_min, "mean", data_max,
            cw=width - 16))

    mean_pos = int((m - data_min) / data_range * (width - 1))
    lower_pos = int((lower - data_min) / data_range * (width - 1))
    upper_pos = int((upper - data_min) / data_range * (width - 1))

    ref = [' '] * width
    ref[max(0, min(width - 1, lower_pos))] = '['
    ref[max(0, min(width - 1, upper_pos))] = ']'
    ref[max(0, min(width - 1, mean_pos))] = '\u2502'
    lines.append("  {:>4}  {:>8}  {}  \u00b12\u03c3 band".format(
        "", "", ''.join(ref)))

    for i, v in enumerate(data):
        pos = int((v - data_min) / data_range * (width - 1))
        pos = max(0, min(width - 1, pos))

        row = [' '] * width
        row[max(0, min(width - 1, mean_pos))] = '\u00b7'
        row[max(0, min(width - 1, lower_pos))] = ':'
        row[max(0, min(width - 1, upper_pos))] = ':'

        outside = v > upper or v < lower
        row[pos] = '\u2716' if outside else '\u25cf'

        lines.append("  {:>4}  {:>8.2f}  {}".format(
            i + 1, v, ''.join(row)))

    lines.append("")
    lines.append(
        "  \u2502 = mean ({:.2f})    : = \u00b12\u03c3 bounds "
        "[{:.2f}, {:.2f}]".format(m, lower, upper))
    lines.append(
        "  \u25cf = within bounds    \u2716 = outside \u00b12\u03c3")

    outside_count = sum(1 for v in data if v > upper or v < lower)
    expected = max(1, int(n * 0.045))
    lines.append(
        "  Outside \u00b12\u03c3: {}/{} "
        "(expected ~{} by chance for n={})".format(
            outside_count, n, expected, n))

    if outside_count > expected * 2:
        lines.append(
            "  \u26a0\ufe0f  More outliers than expected "
            "\u2014 possible instability")
    else:
        lines.append("  \u2705 Within expected variation")
    lines.append("")
    return '\n'.join(lines)


# =============================================================================
# Cumulative Comparison
# =============================================================================

def cumulative_comparison(data1, data2, label1, label2):
    """Generate cumulative threshold comparison table.

    Complexity: O((n1+n2) log(n1+n2)) for sorting + O(t + log(n)) for vectorized searchsorted,
    where n1, n2 are dataset sizes and t is number of thresholds.
    Replaces the prior per-threshold lookup loop with a vectorized
    np.searchsorted pass.
    """
    all_arr = np.concatenate([np.asarray(data1), np.asarray(data2)])
    m = float(np.mean(all_arr))
    s = float(np.std(all_arr, ddof=1))

    if s == 0:
        lines = []
        lines.append("  Percentage of values \u2265 threshold:")
        lines.append("")
        lines.append("  All values are identical ({:.2f}) "
                      "\u2014 cumulative comparison not applicable".format(m))
        lines.append("")
        return '\n'.join(lines)

    step = s / 2
    start = int(round((m - 2 * s) / 5)) * 5
    thresholds = []
    t = start
    while t <= m + 2 * s:
        thresholds.append(int(round(t)))
        t += step

    thresholds = sorted(set(thresholds))

    # Sort datasets once and vectorize threshold lookups with np.searchsorted
    sorted1 = np.sort(data1)
    sorted2 = np.sort(data2)
    n1, n2 = len(sorted1), len(sorted2)

    # Vectorized threshold lookup: compute counts for all thresholds at once
    thresholds_arr = np.array(thresholds)
    counts1 = n1 - np.searchsorted(sorted1, thresholds_arr, side='left')
    counts2 = n2 - np.searchsorted(sorted2, thresholds_arr, side='left')

    lines = []
    lines.append("  Percentage of values \u2265 threshold:")
    lines.append("")
    col_w = max(10, len(label1), len(label2))
    lines.append(
        "  {:>10}  {:>{w}}  {:>{w}}  {:>10}  {:>{w}}".format(
            "Threshold", label1, label2, "Difference", "Better", w=col_w))
    lines.append(
        "  {}  {}  {}  {}  {}".format(
            "\u2500" * 10, "\u2500" * col_w,
            "\u2500" * col_w, "\u2500" * 10, "\u2500" * col_w))

    for i, t in enumerate(thresholds):
        count1 = int(counts1[i])
        count2 = int(counts2[i])
        if count1 == 0 and count2 == 0:
            continue
        pct1 = 100.0 * count1 / n1
        pct2 = 100.0 * count2 / n2
        diff = pct2 - pct1
        better = (label2 if diff > 2
                  else (label1 if diff < -2 else "\u2248 tie"))
        lines.append(
            "  {:>10}  {:>{w}.0f}%  {:>{w}.0f}%  {:>+9.0f}%  {:>{w}}".format(
                "\u2265 {}".format(t), pct1, pct2, diff, better, w=col_w - 1))

    lines.append("")
    return '\n'.join(lines)


# =============================================================================
# Report Helpers
# =============================================================================

def significance_marker(p, alpha=0.05):
    if p < alpha / 10:
        return "\u2705 Yes (highly)"
    elif p < alpha:
        return "\u2705 Yes"
    elif p < alpha * 2:
        return "\u26a0\ufe0f  Borderline"
    else:
        return "\u274c No"


def interpret_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def one_tailed_p_value(p_two, signed_effect, higher_is_better):
    """Convert a two-tailed p-value to one-tailed, respecting hypothesis direction.

    signed_effect should be positive when experiment > baseline.
    """
    p_two = max(0.0, min(1.0, p_two))
    if signed_effect == 0:
        return 0.5

    if higher_is_better:
        return p_two / 2.0 if signed_effect > 0 else 1.0 - p_two / 2.0
    return p_two / 2.0 if signed_effect < 0 else 1.0 - p_two / 2.0


def _append_report_section(report, title):
    """Append a standard section header used throughout reports."""
    report.append("\u2500" * 72)
    report.append("  {}".format(title))
    report.append("\u2500" * 72)


# =============================================================================
# Single Experiment Report
# =============================================================================

def generate_single_report(data, label, higher_is_better=True, alpha=0.05):
    """Generate analysis report for a single experiment."""
    report = []
    stats = compute_descriptive_stats(data)

    report.append("=" * 72)
    report.append("  BENCHMARK ANALYSIS REPORT")
    report.append("  {}".format(label))
    report.append("  {}".format(
        "Higher is better" if higher_is_better else "Lower is better"))
    report.append("=" * 72)
    report.append("")

    # -- Descriptive Statistics --
    _append_report_section(report, "DESCRIPTIVE STATISTICS")

    stat_lines, ci_lo, ci_hi, ci_fmt = _format_stats_single(stats, alpha)
    report.extend(stat_lines)
    report.append("")

    m = stats['mean']
    s = stats['stdev']
    n = stats['n']
    se = stats['se']
    sk = stats['skewness']
    ek = stats['ex_kurtosis']

    cv = 100.0 * s / abs(m) if m != 0 else float('inf')
    if m != 0:
        report.append("  Coefficient of Variation: {:.2f}%".format(cv))
        report.append(interpret_cv(cv))
    else:
        report.append("  Coefficient of Variation: N/A (mean is zero)")
    report.append("")

    # -- Histogram --
    _append_report_section(report, "FREQUENCY HISTOGRAM")

    edges = _compute_bins([data])
    report.append(_single_histogram(data, label, edges, stats=stats))

    # -- Distribution Shape --
    _append_report_section(report, "DISTRIBUTION SHAPE")
    report.append(_distribution_shape_comparison([data], [label], edges))
    report.append("")

    # -- Normality --
    _append_report_section(report, "NORMALITY ASSESSMENT")

    norm_lines, p_values = _normality_assessment(
        [data], [label], edges, alpha, stats_list=[stats])
    report.extend(norm_lines)
    p_norm = p_values[0]

    report.extend(_interpret_skewness(sk))
    report.extend(_interpret_kurtosis(ek))
    report.append("")

    # -- Outlier Detection --
    _append_report_section(report, "OUTLIER DETECTION")

    outliers_iqr = detect_outliers_iqr(
        data, stats['p25'], stats['p75'], stats['iqr'])
    outliers_zscore = detect_outliers_zscore(
        data, stats['mean'], stats['stdev'])
    report.extend(format_outlier_report(stats, outliers_iqr, outliers_zscore))

    # -- Stability --
    _append_report_section(report, "STABILITY ANALYSIS")
    report.append(_stability_chart(data, label, stats=stats))

    # -- Bootstrap CI --
    _append_report_section(report, "BOOTSTRAP CONFIDENCE INTERVAL FOR MEAN")

    boot_lo, boot_hi = bootstrap_single_ci(
        data, ci=int(100 * (1 - alpha)))

    report.append(
        "  {:.0f}% Bootstrap CI: [{:.2f}, {:.2f}]".format(
            100 * (1 - alpha), boot_lo, boot_hi))
    report.append(
        "  Parametric CI:   {}".format(ci_fmt))
    report.append("")
    agreement = abs(boot_lo - ci_lo) <= se and abs(boot_hi - ci_hi) <= se
    if agreement:
        report.append(
            "  \u2705 Bootstrap and parametric CIs agree "
            "\u2014 estimates are reliable")
    else:
        report.append(
            "  \u26a0\ufe0f  Bootstrap and parametric CIs diverge \u2014 "
            "distribution may be non-normal")
    report.append("")

    # -- Sample Size --
    _append_report_section(report, "SAMPLE SIZE ADEQUACY")
    # Reuse already computed parametric CI to avoid redoing t-critical × SE work.
    moe = abs(ci_hi - m)
    report.append(_sample_size_analysis_single(
        stats, alpha, moe=moe, ci_lo=ci_lo, ci_hi=ci_hi))

    # -- Summary --
    _append_report_section(report, "SUMMARY")

    report.append("  Benchmark: {}".format(label))
    report.append("  Result:    {:.2f} \u00b1 {:.2f} ops/min (mean \u00b1 std)".format(
        m, s))
    report.append(
        "  {:.0f}% CI:    {} ops/min".format(
            100 * (1 - alpha), ci_fmt))
    if m != 0:
        report.append("  CV:        {:.2f}%".format(cv))
    else:
        report.append("  CV:        N/A (mean is zero)")
    report.append("  Outliers:  {} (IQR method)".format(len(outliers_iqr)))
    report.append("  Normal:    {} (p = {:.4f})".format(
        "Yes" if p_norm > alpha else "No", p_norm))
    report.append("")

    report.append("  Recommendations:")
    if n < 30:
        report.append(
            "  \u26a0\ufe0f  Sample size < 30 \u2014 consider collecting "
            "more data for reliable statistics")
    if m != 0 and cv > 5:
        report.append(
            "  \u26a0\ufe0f  High variability \u2014 "
            "investigate sources of noise")
    if len(outliers_iqr) > 0:
        report.append(
            "  \u26a0\ufe0f  {} outlier(s) detected \u2014 "
            "consider investigating whether they represent real behavior "
            "or measurement errors".format(len(outliers_iqr)))
    if p_norm <= alpha:
        report.append(
            "  \u26a0\ufe0f  Non-normal distribution \u2014 "
            "use non-parametric methods for comparisons")
    if (n >= 30 and cv <= 5 and
            len(outliers_iqr) == 0 and p_norm > alpha):
        report.append("  \u2705 Data looks clean and well-behaved")
    report.append("")
    report.append("=" * 72)

    return '\n'.join(report)


# =============================================================================
# Comparison Report
# =============================================================================

def _format_diff_pct(diff_pct):
    """Format a percentage difference, handling inf gracefully."""
    if math.isinf(diff_pct):
        return "N/A (baseline mean is zero)"
    return "{:+.2f}%".format(diff_pct)


def _compute_p_value_scale_info(alpha, max_p=0.0):
    """Compute dynamic p-value visual scale based on alpha and observed p-values.

    Args:
        alpha: Significance level
        max_p: Maximum observed p-value being plotted (default 0.0 for backward compat)

    Returns (scale_header_label, tick_bar_label, position_multiplier, scale_width):
    - scale_header_label: formatted header label (e.g., "0.00  0.01  0.02  ...")
    - tick_bar_label: formatted tick bar (e.g., "|     |     |     ...")
    - position_multiplier: factor to multiply p-value by to get position
    - scale_width: actual character width of the tick scale
    """
    # Choose candidate scale based on alpha
    if alpha <= 0.05:
        scale_max = 0.05
    elif alpha <= 0.10:
        scale_max = 0.10
    elif alpha <= 0.20:
        scale_max = 0.20
    else:
        scale_max = 0.50

    # Widen scale if any observed p-value exceeds it
    if max_p > scale_max:
        # Round up to next clean boundary
        candidates = [0.05, 0.10, 0.20, 0.50, 1.0]
        for c in candidates:
            if c >= max_p:
                scale_max = c
                break
        else:
            scale_max = 1.0

    # Compute tick values: ~6 evenly spaced ticks from 0 to scale_max
    n_ticks = 6
    tick_step = scale_max / (n_ticks - 1)
    tick_values = [i * tick_step for i in range(n_ticks)]

    tick_labels = "  ".join("{:.2f}".format(t) for t in tick_values)
    tick_bar = "\u2502" + "     \u2502" * (n_ticks - 1)

    # Compute actual scale width: tick labels are 4 chars wide, joined by 2 spaces
    # So each tick is 6 positions apart: (n_ticks - 1) * 6 = scale_width
    scale_width = (len(tick_values) - 1) * 6
    position_multiplier = scale_width / scale_max

    return tick_labels, tick_bar, position_multiplier, scale_width


def generate_report(data_base, data_exp, label_base, label_exp,
                    higher_is_better=True, alpha=0.05):
    """Generate the full comparison report."""
    report = []
    stats_base = compute_descriptive_stats(data_base)
    stats_exp = compute_descriptive_stats(data_exp)

    report.append("=" * 72)
    report.append("  BENCHMARK COMPARISON REPORT")
    report.append(
        "  {} (baseline) vs. {} (experiment)".format(
            label_base, label_exp))
    report.append("  {}".format(
        "Higher is better" if higher_is_better else "Lower is better"))
    report.append("=" * 72)
    report.append("")

    # -- Descriptive Statistics --
    _append_report_section(report, "DESCRIPTIVE STATISTICS")

    report.extend(_format_stats_comparison(
        stats_base, stats_exp, label_base, label_exp))

    diff_mean = stats_exp['mean'] - stats_base['mean']
    s_pool = _pooled_stdev_from_params(
        stats_base['n'], stats_base['var'],
        stats_exp['n'], stats_exp['var'])
    d = (diff_mean / s_pool if s_pool != 0.0 else 0.0)

    if stats_base['mean'] != 0:
        diff_pct = 100.0 * diff_mean / stats_base['mean']
    else:
        diff_pct = float('inf') if diff_mean != 0 else 0.0
    # Compute diff_pct formatted string once, reuse in VERDICT section
    diff_pct_str = _format_diff_pct(diff_pct)
    report.append("")
    report.append(
        "  Observed difference: {:+.2f} ({})".format(
            diff_mean, diff_pct_str))

    # Handle zero difference case separately (neither improvement nor regression)
    if diff_mean == 0:
        direction_good = False
        report.append("  Direction: \u2796 No change")
    else:
        direction_good = (diff_mean > 0) == higher_is_better
        report.append(
            "  Direction: {}".format(
                "\u2705 Improvement" if direction_good
                else "\u274c Regression"))
    report.append("")

    # -- Histograms --
    _append_report_section(report, "FREQUENCY HISTOGRAMS")

    edges = _compute_bins([data_base, data_exp])

    report.append(_single_histogram(data_base, label_base, edges, stats=stats_base))
    report.append(_single_histogram(data_exp, label_exp, edges, stats=stats_exp))

    # -- Overlay --
    _append_report_section(report, "OVERLAY HISTOGRAM")
    report.append(overlay_histogram(
        data_base, data_exp, label_base, label_exp, edges))

    # -- Distribution Shape --
    _append_report_section(report, "DISTRIBUTION SHAPE COMPARISON")
    report.append(_distribution_shape_comparison(
        [data_base, data_exp],
        [label_base, label_exp],
        edges))
    report.append("")

    # -- Normality --
    _append_report_section(report, "NORMALITY ASSESSMENT")

    norm_lines, _ = _normality_assessment(
        [data_base, data_exp],
        [label_base, label_exp],
        edges, alpha, stats_list=[stats_base, stats_exp])
    report.extend(norm_lines)

    # -- Statistical Tests --
    _append_report_section(report, "STATISTICAL TESTS")

    if higher_is_better:
        one_tail_label = "{} > {}".format(label_exp, label_base)
    else:
        one_tail_label = "{} < {}".format(label_exp, label_base)

    report.append("  Hypothesis: {}".format(one_tail_label))
    report.append("  Significance level: \u03b1 = {}".format(alpha))
    report.append("")

    # Welch's t-test
    t_stat, df, p_two = welch_t_test(data_base, data_exp)
    p_one = one_tailed_p_value(p_two, t_stat, higher_is_better)

    # Mann-Whitney
    u_stat, z_stat, p_two_mw = mann_whitney_u(data_base, data_exp)
    p_one_mw = one_tailed_p_value(p_two_mw, z_stat, higher_is_better)

    # Permutation
    perm_diff, p_two_perm = permutation_test(data_base, data_exp)
    p_one_perm = one_tailed_p_value(p_two_perm, perm_diff, higher_is_better)

    # Bootstrap
    boot_lo, boot_hi, p_two_boot = bootstrap_ci(data_base, data_exp)
    p_one_boot = one_tailed_p_value(p_two_boot, diff_mean, higher_is_better)

    test_results = [
        ("Welch's t-test",
         "t={:.3f}, df={:.1f}".format(t_stat, df),
         p_one, p_two),
        ("Mann-Whitney U",
         "U={:.0f}, z={:.3f}".format(u_stat, z_stat),
         p_one_mw, p_two_mw),
        ("Permutation test",
         "\u03b4={:.2f}".format(perm_diff),
         p_one_perm, p_two_perm),
        ("Bootstrap",
         "CI=[{:.2f}, {:.2f}]".format(boot_lo, boot_hi),
         p_one_boot, p_two_boot),
    ]

    report.append(
        "  {:<20}  {:<25}  {:>10}  {:>10}  {:>18}".format(
            "Test", "Statistic", "p (1-tail)",
            "p (2-tail)", "Significant?"))
    report.append(
        "  {}  {}  {}  {}  {}".format(
            "\u2500" * 20, "\u2500" * 25,
            "\u2500" * 10, "\u2500" * 10, "\u2500" * 18))

    for name, stat_str, p1, p2 in test_results:
        sig = significance_marker(p1, alpha)
        report.append(
            "  {:<20}  {:<25}  {:>10.4f}  {:>10.4f}  {:>18}".format(
                name, stat_str, p1, p2, sig))

    report.append("")
    # Compute Cohen's d summary once, reuse in VERDICT section
    d_summary = "Cohen's d: {:.3f} ({})".format(d, interpret_cohens_d(d))
    report.append("  {}".format(d_summary))
    report.append(
        "    Interpretation: The means are {:.2f} pooled "
        "standard deviations apart.".format(abs(d)))

    if abs(d) > 0:
        win_prob = float(sp_stats.norm.cdf(d / math.sqrt(2)))
        report.append(
            "    Win probability: A random {} value beats "
            "a random {} value ~{:.0f}% of the time.".format(
                label_exp, label_base, win_prob * 100))
    report.append("")

    # -- p-value visual --
    report.append("  p-value visual summary (one-tailed):")
    report.append("")

    # Compute dynamic scale based on alpha and max observed p-value
    max_p = max(p1 for _, _, p1, _ in test_results)
    tick_labels, tick_bar, pos_mult, scale_width = _compute_p_value_scale_info(alpha, max_p)

    report.append(
        "  {:<20}  {:>7}  "
        "{}".format("Test", "p-value", tick_labels))
    report.append(
        "  {:20}  {:7}  "
        "{}".format("", "", tick_bar))

    for name, _, p1, _ in test_results:
        pos = min(int(p1 * pos_mult), scale_width)
        bar = ' ' * pos + '\u2593'
        report.append(
            "  {:<20}  {:>7.4f}  \u2502{}".format(name, p1, bar))

    # Format alpha label line with dynamic positioning based on scale
    alpha_pos = min(int(alpha * pos_mult), scale_width)
    alpha_line = "  {:20}  {:7}  \u2502{}\u03b1={:.4f}".format(
        "", "", " " * alpha_pos, alpha)
    report.append(alpha_line)
    report.append("")

    # -- Sample Size & Power --
    _append_report_section(report, "SAMPLE SIZE & POWER ANALYSIS")
    report.append(_sample_size_analysis_comparison(
        stats_base, stats_exp, label_base, label_exp, alpha,
        s_pool=s_pool, observed_d=d))

    # -- Cumulative comparison --
    _append_report_section(report, "CUMULATIVE COMPARISON")
    report.append(cumulative_comparison(
        data_base, data_exp, label_base, label_exp))

    # -- Verdict --
    _append_report_section(report, "VERDICT")

    sig_count = sum(1 for _, _, p1, _ in test_results if p1 < alpha)
    total_tests = len(test_results)

    report.append(
        "  Tests significant at \u03b1={}: {}/{}".format(
            alpha, sig_count, total_tests))
    report.append(
        "  Mean difference: {:+.2f} ({})".format(
            diff_mean, diff_pct_str))
    report.append(
        "  {}".format(d_summary))
    report.append(
        "  Bootstrap 95% CI: [{:.2f}, {:.2f}]".format(boot_lo, boot_hi))
    report.append("")

    if sig_count >= total_tests * 0.75:
        if direction_good:
            verdict = (
                "  \u2705 SIGNIFICANT IMPROVEMENT: {} is "
                "significantly better than {}.".format(
                    label_exp, label_base))
        else:
            verdict = (
                "  \u274c SIGNIFICANT REGRESSION: {} is "
                "significantly worse than {}.".format(
                    label_exp, label_base))
    elif sig_count > 0:
        verdict = (
            "  \u26a0\ufe0f  MIXED RESULTS: Some tests show significance, "
            "but results are not fully consistent.")
    else:
        verdict = (
            "  \u2796 NO SIGNIFICANT DIFFERENCE between "
            "{} and {}.".format(label_base, label_exp))

    report.append(verdict)

    report.append("")
    if abs(d) < 0.2:
        report.append(
            "  Practical significance: NEGLIGIBLE \u2014 "
            "difference is too small to matter.")
    elif abs(d) < 0.5:
        report.append(
            "  Practical significance: SMALL \u2014 real but modest "
            "improvement. Consider code quality and other factors.")
    elif abs(d) < 0.8:
        report.append(
            "  Practical significance: MEDIUM \u2014 "
            "noticeable improvement.")
    else:
        report.append(
            "  Practical significance: LARGE \u2014 "
            "substantial improvement.")

    report.append("")
    report.append("=" * 72)

    return '\n'.join(report)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results. "
                    "One file: single analysis. "
                    "Two files: comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.csv
  %(prog)s latency_base.csv latency_fix.csv --lower-is-better
  %(prog)s base.csv fix.csv --alpha 0.01 --base-label "Baseline" --exp-label "Fix"
        """)
    parser.add_argument(
        "baseline",
        help="Path to baseline/single CSV file")
    parser.add_argument(
        "experiment",
        nargs='?',
        default=None,
        help="Path to experiment CSV file (optional)")
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        default=False,
        help="Lower values mean better performance")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)")
    parser.add_argument(
        "--base-label",
        default=None,
        help="Label for baseline (default: filename)")
    parser.add_argument(
        "--exp-label",
        default=None,
        help="Label for experiment (default: filename)")

    args = parser.parse_args()

    higher_is_better = not args.lower_is_better

    data_base = load_csv(args.baseline)
    if len(data_base) < 3:
        print("Error: Need at least 3 data points in {} (got {})".format(
            args.baseline, len(data_base)),
              file=sys.stderr)
        sys.exit(1)

    label_base = (args.base_label or
                  args.baseline.replace('.csv', '').split('/')[-1])

    if args.experiment is None:
        # -- Single experiment analysis --
        report = generate_single_report(
            data_base, label_base,
            higher_is_better=higher_is_better,
            alpha=args.alpha)
    else:
        # -- Two experiment comparison --
        data_exp = load_csv(args.experiment)
        if len(data_exp) < 3:
            print("Error: Need at least 3 data points in {} (got {})".format(
                args.experiment, len(data_exp)),
                  file=sys.stderr)
            sys.exit(1)

        label_exp = (args.exp_label or
                     args.experiment.replace('.csv', '').split('/')[-1])

        report = generate_report(
            data_base, data_exp,
            label_base, label_exp,
            higher_is_better=higher_is_better,
            alpha=args.alpha)

    print(report)


if __name__ == "__main__":
    main()
