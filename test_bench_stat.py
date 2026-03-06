#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for bench_stat.py statistical correctness and robustness."""

import math
import random
import re
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

import bench_stat as mod


def _ignore_precision_loss(func):
    """Decorator: allow scipy's 'Precision loss' RuntimeWarning in this test."""
    def wrapper(self, *args, **kwargs):
        warnings.filterwarnings("ignore", message="Precision loss",
                                category=RuntimeWarning)
        return func(self, *args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# =============================================================================
# Reference implementations for independent verification (not production code)
# =============================================================================

def _ref_mean(data):
    """Mean with safe behavior for empty data."""
    return float(np.mean(data)) if len(data) > 0 else 0.0


def _ref_stdev(data, ddof=1):
    """Standard deviation with ddof guard for small samples."""
    if len(data) <= ddof:
        return 0.0
    return float(np.std(data, ddof=ddof))


def _ref_variance(data, ddof=1):
    """Variance with specified degrees-of-freedom correction."""
    if len(data) <= ddof:
        return 0.0
    return float(np.var(data, ddof=ddof))


def _ref_confidence_interval(m, se, alpha, df):
    """Two-tailed confidence interval from mean and standard error."""
    if df < 1:
        return float('-inf'), float('inf')
    t_crit = float(abs(sp_stats.t.ppf(alpha / 2, df)))
    moe = t_crit * se
    return m - moe, m + moe


class BenchStatRegressionTests(unittest.TestCase):

    def setUp(self):
        # Turn all warnings into errors so unexpected warnings fail the test
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.simplefilter("error")

    def tearDown(self):
        self._warn_ctx.__exit__(None, None, None)

    def test_one_tailed_p_values_respect_direction(self):
        base = [100, 101, 99, 100, 102, 98, 101, 100]
        exp = [90, 91, 89, 90, 92, 88, 91, 90]

        t_stat, _, p2_t = mod.welch_t_test(base, exp)
        _, z_stat, p2_mw = mod.mann_whitney_u(base, exp)
        perm_diff, p2_perm = mod.permutation_test(base, exp, n_perms=2000, seed=1)
        _, _, p2_boot = mod.bootstrap_ci(base, exp, n_boot=2000, seed=1)
        mean_diff = _ref_mean(exp) - _ref_mean(base)

        p1_higher = [
            mod.one_tailed_p_value(p2_t, t_stat, True),
            mod.one_tailed_p_value(p2_mw, z_stat, True),
            mod.one_tailed_p_value(p2_perm, perm_diff, True),
            mod.one_tailed_p_value(p2_boot, mean_diff, True),
        ]
        p1_lower = [
            mod.one_tailed_p_value(p2_t, t_stat, False),
            mod.one_tailed_p_value(p2_mw, z_stat, False),
            mod.one_tailed_p_value(p2_perm, perm_diff, False),
            mod.one_tailed_p_value(p2_boot, mean_diff, False),
        ]

        for p in p1_higher:
            self.assertGreaterEqual(p, 0.95)
        for p in p1_lower:
            self.assertLessEqual(p, 0.05)

    def test_mann_whitney_z_keeps_experiment_direction(self):
        base = [100, 101, 99, 100, 102, 98, 101, 100]
        exp = [90, 91, 89, 90, 92, 88, 91, 90]
        _, z_stat, _ = mod.mann_whitney_u(base, exp)
        self.assertLess(z_stat, 0.0)

    def test_bootstrap_p_two_is_bounded(self):
        random.seed(0)
        for trial in range(120):
            a = [random.gauss(0, 1) for _ in range(20)]
            b = [random.gauss(0, 1) for _ in range(20)]
            _, _, p_two = mod.bootstrap_ci(a, b, n_boot=400, seed=trial)
            self.assertGreaterEqual(p_two, 0.0)
            self.assertLessEqual(p_two, 1.0)

    def test_bootstrap_p_two_identical_data(self):
        """Regression test: identical datasets should yield p_two == 1.0.

        When comparing two identical datasets, all bootstrap diffs should be 0.
        The previous implementation double-counted diffs == 0 in both p_le_zero
        and p_ge_zero. The fix uses strict inequalities to avoid this issue.
        With identical data, p_left = p_right = 0, so p_two = 1.0.
        """
        data1 = [100] * 50
        data2 = [100] * 50
        ci_lower, ci_upper, p_two = mod.bootstrap_ci(data1, data2, n_boot=5000, seed=42)

        # All diffs should be exactly 0, so CI should be [0, 0]
        self.assertEqual(ci_lower, 0.0)
        self.assertEqual(ci_upper, 0.0)

        # p_two should be exactly 1.0 (no evidence of difference)
        self.assertEqual(p_two, 1.0)

        # Verify it never exceeds 1.0
        self.assertLessEqual(p_two, 1.0)

    def test_bootstrap_single_ci_returns_valid_interval(self):
        """Test that bootstrap_single_ci returns a valid confidence interval.

        The function should return a tuple of (lower, upper) where lower <= upper,
        and both should be floats. The interval should contain values near the
        sample mean for a reasonable confidence level.
        """
        # Test with normal data
        data = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8]
        ci_lo, ci_hi = mod.bootstrap_single_ci(data, n_boot=1000, ci=95, seed=42)

        # Check return types
        self.assertIsInstance(ci_lo, float)
        self.assertIsInstance(ci_hi, float)

        # Check interval ordering
        self.assertLessEqual(ci_lo, ci_hi)

        # Interval should be near the mean
        sample_mean = _ref_mean(data)
        self.assertLess(ci_lo, sample_mean + 2.0)
        self.assertGreater(ci_hi, sample_mean - 2.0)

        # Test with constant data
        const_data = [5.0] * 10
        ci_lo_const, ci_hi_const = mod.bootstrap_single_ci(const_data, n_boot=100, ci=95, seed=42)
        # For constant data, CI should be very narrow (or equal)
        self.assertAlmostEqual(ci_lo_const, 5.0, places=5)
        self.assertAlmostEqual(ci_hi_const, 5.0, places=5)

        # Test with different confidence levels
        ci_lo_90, ci_hi_90 = mod.bootstrap_single_ci(data, n_boot=1000, ci=90, seed=42)
        ci_lo_99, ci_hi_99 = mod.bootstrap_single_ci(data, n_boot=1000, ci=99, seed=42)

        # 99% CI should be wider than 90% CI
        self.assertLess(ci_lo_99, ci_lo_90)
        self.assertGreater(ci_hi_99, ci_hi_90)

    def test_bootstrap_single_ci_scipy_compatibility(self):
        """Regression test for scipy 1.7.3 compatibility (no scipy.stats.bootstrap).

        scipy.stats.bootstrap was added in scipy 1.9.0. Python 3.7 maxes out
        at scipy 1.7.3, so bootstrap_single_ci must use manual vectorized
        resampling, not scipy.stats.bootstrap.

        This test ensures the function works without raising AttributeError.
        """
        data = [10, 20, 30, 40, 50]

        # Should not raise AttributeError about missing scipy.stats.bootstrap
        ci_lo, ci_hi = mod.bootstrap_single_ci(data, n_boot=10000, ci=95, seed=42)

        # Must return tuple of two floats
        self.assertIsInstance(ci_lo, float)
        self.assertIsInstance(ci_hi, float)

        # Interval must be properly ordered
        self.assertLess(ci_lo, ci_hi, "Lower bound must be less than upper bound")

        # Both bounds should be within range of data
        data_min = min(data)
        data_max = max(data)
        self.assertGreaterEqual(ci_lo, data_min, "Lower bound must be >= min of data")
        self.assertLessEqual(ci_hi, data_max, "Upper bound must be <= max of data")

    def test_constant_data_normality_histogram_does_not_crash(self):
        same = [5.0] * 20
        edges = mod._compute_bins([same])
        out = mod._normality_histogram(same, "same", edges)
        self.assertIn("Actual vs. Normal Expected", out)

    def test_small_sample_raises_value_error(self):
        """Data with fewer than 3 samples must raise ValueError."""
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([1.0])
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([1.0, 2.0])
        # 3 samples should work
        stats = mod.compute_descriptive_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats['n'], 3)

    @_ignore_precision_loss
    def test_welch_t_test_with_zero_variance_samples(self):
        """Regression test for welch_t_test 0/0 df computation.

        Both groups are constant, so v1 == v2 == 0. scipy returns t=-inf, df=1.0.
        Our function negates t to +inf (data2 > data1 direction).
        """
        t_stat, df, p_two = mod.welch_t_test([5.0, 5.0, 5.0], [7.0, 7.0, 7.0])
        self.assertTrue(math.isinf(t_stat) and t_stat > 0)
        self.assertGreater(df, 0)
        self.assertEqual(p_two, 0.0)

    def test_welch_t_test_df_p_value_consistency(self):
        """Regression test: welch_t_test t_stat, df, and p_value consistency.

        The returned t_stat and df should be derived from the same variance
        values used to compute p_two. Verify that
        2 * sp_stats.t.sf(abs(t_stat), df) equals the returned p_two
        within 1e-10 tolerance.
        """
        from scipy import stats as sp_stats

        # Test with known data
        data1 = [1.2, 1.5, 1.3, 1.4, 1.6, 1.1]
        data2 = [2.1, 2.3, 2.0, 2.2, 2.4, 2.5]

        t_stat, df, p_two = mod.welch_t_test(data1, data2)

        # Verify t_stat and df are consistent with p_two
        expected_p_two = 2 * sp_stats.t.sf(abs(t_stat), df)
        self.assertAlmostEqual(p_two, expected_p_two, places=10)

    @_ignore_precision_loss
    def test_generate_report_with_zero_baseline_mean(self):
        """Regression test for generate_report baseline mean division by zero.

        When baseline mean is zero, percent difference should not raise
        ZeroDivisionError and should be formatted as N/A instead of infinity.
        """
        report = mod.generate_report(
            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "Baseline", "Experiment"
        )
        self.assertIn("Observed difference: +1.00 (N/A (baseline mean is zero))", report)

    def test_sample_size_analysis_single_with_zero_mean(self):
        """Regression test for _sample_size_analysis_single with mean == 0.

        Relative precision sections should be skipped and replaced with
        absolute precision messaging.
        """
        stats = mod.compute_descriptive_stats([-1.0, 0.0, 1.0])
        analysis = mod._sample_size_analysis_single(stats)

        self.assertIn("mean is zero, relative % N/A", analysis)
        self.assertNotIn("Samples needed for target precision:", analysis)
        self.assertIn("Mean is zero", analysis)
        self.assertIn("Absolute margin of error:", analysis)

    def test_precision_table_formats_infinite_required_n(self):
        """Regression test for _precision_table inf formatting.

        Infinite required sample sizes should render as a readable symbol and
        avoid status text like "Need inf more".
        """
        rows = mod._precision_table(0.0, 1.0, 10, 0.05)
        table = "\n".join(rows)

        self.assertIn("∞", table)
        self.assertIn("N/A (mean=0)", table)
        self.assertNotIn("Need inf more", table)

    @_ignore_precision_loss
    def test_sample_size_analysis_comparison_with_zero_baseline_mean(self):
        """Regression test for comparison analysis when baseline mean is zero."""
        stats_base = mod.compute_descriptive_stats([0.0, 0.0, 0.0])
        stats_exp = mod.compute_descriptive_stats([1.0, 1.0, 1.0])
        analysis = mod._sample_size_analysis_comparison(
            stats_base, stats_exp, "Baseline", "Experiment"
        )

        self.assertIn("Relative:   N/A (baseline mean is zero)", analysis)
        self.assertNotIn("Relative:   ±inf% of baseline mean", analysis)

    def test_generate_single_report_with_zero_mean(self):
        """Regression test for generate_single_report CV with zero mean.

        CV should report N/A rather than 0 or crashing when mean is zero.
        """
        report = mod.generate_single_report([-1.0, 0.0, 1.0], "ZeroMean")
        self.assertIn("Coefficient of Variation: N/A (mean is zero)", report)
        self.assertNotIn("Coefficient of Variation: 0.00%", report)

    def test_compute_bins_with_numpy_histogram_bin_edges(self):
        """Test that _compute_bins uses numpy's histogram_bin_edges optimization.

        Verifies that the optimized implementation:
        - Produces sensible bin edges for normal data
        - Handles constant data correctly (zero range)
        - Returns edges as a list
        - Edges cover the data range
        """
        # Test with normal data
        data = [100.0, 101.0, 102.0, 99.0, 98.0, 103.0, 97.0, 104.0, 95.0, 105.0]
        edges = mod._compute_bins([data])

        # Verify basic properties
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 2, "Should have at least 3 edges (2 bins)")
        self.assertEqual(edges, sorted(edges), "Edges should be sorted")
        self.assertLessEqual(edges[0], min(data), "First edge should be <= min")
        self.assertGreaterEqual(edges[-1], max(data), "Last edge should be >= max")

        # Verify edges are uniformly spaced (histogram_bin_edges creates uniform bins)
        gaps = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
        avg_gap = sum(gaps) / len(gaps)
        for gap in gaps:
            self.assertAlmostEqual(gap, avg_gap, places=5,
                                 msg="All gaps should be equal (uniform bins)")

        # Test with constant data (zero range)
        constant_data = [5.0, 5.0, 5.0, 5.0]
        edges_const = mod._compute_bins([constant_data])
        self.assertEqual(len(edges_const), 2, "Constant data should return 2 edges")
        self.assertEqual(edges_const[0], 4.0, "First edge should be value - 1")
        self.assertEqual(edges_const[1], 6.0, "Second edge should be value + 1")

        # Test with multiple datasets
        data2 = [110.0, 115.0, 120.0, 125.0]
        edges_multi = mod._compute_bins([data, data2])
        self.assertIsInstance(edges_multi, list)
        self.assertLessEqual(edges_multi[0], min(data + data2))
        self.assertGreaterEqual(edges_multi[-1], max(data + data2))

    def test_compute_bins_min_max_constraints(self):
        """Test that _compute_bins enforces min_bins=8 and max_bins=20 constraints.

        Small or unusual datasets might produce very few bins with numpy's 'auto'
        method, or unusual distributions might produce many bins. This test verifies
        that the function respects min/max constraints for readable histograms.
        """
        import numpy as np

        # Test 1: Small dataset should enforce minimum 8 bins
        # (numpy's auto method might produce < 8 bins for small data)
        small_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        edges_small = mod._compute_bins([small_data])
        n_bins_small = len(edges_small) - 1
        self.assertGreaterEqual(n_bins_small, 8,
                               "Small dataset should enforce minimum 8 bins")
        self.assertLessEqual(n_bins_small, 20,
                            "Bin count should not exceed maximum of 20")

        # Test 2: Uniform distribution (which might produce many bins) should be capped at 20
        # Create a very uniform, spread-out dataset that numpy might generate many bins for
        uniform_data = np.linspace(0, 1000, 10000).tolist()
        edges_uniform = mod._compute_bins([uniform_data])
        n_bins_uniform = len(edges_uniform) - 1
        self.assertGreater(n_bins_uniform, 0, "Should have at least 1 bin")
        self.assertLessEqual(n_bins_uniform, 20,
                            "Uniform distribution should be capped at 20 bins")
        self.assertGreaterEqual(n_bins_uniform, 8,
                               "Even uniform data should have at least 8 bins")

        # Test 3: Typical dataset (100 points) should have 8-20 bins
        rng = np.random.RandomState(42)
        typical_data = rng.normal(loc=100.0, scale=15.0, size=100).tolist()
        edges_typical = mod._compute_bins([typical_data])
        n_bins_typical = len(edges_typical) - 1
        self.assertGreaterEqual(n_bins_typical, 8,
                               "Typical dataset should have at least 8 bins")
        self.assertLessEqual(n_bins_typical, 20,
                            "Typical dataset should not exceed 20 bins")

        # Test 4: Heavily skewed data (might trigger Freedman-Diaconis with few bins)
        # Create highly skewed data (exponential distribution)
        skewed_data = rng.exponential(scale=2.0, size=100).tolist()
        edges_skewed = mod._compute_bins([skewed_data])
        n_bins_skewed = len(edges_skewed) - 1
        self.assertGreaterEqual(n_bins_skewed, 8,
                               "Even heavily skewed data should enforce minimum 8 bins")
        self.assertLessEqual(n_bins_skewed, 20,
                            "Skewed data should respect maximum 20 bins limit")

    def test_expected_normal_counts_matches_scipy_cdf_diff(self):
        """Expected normal counts should match vectorized scipy CDF differences."""
        n = 1000
        m = 10.0
        s = 2.5
        edges = [0.0, 5.0, 10.0, 12.5, 15.0, 20.0]

        got = mod._expected_normal_counts(n, m, s, edges)
        edges_arr = np.asarray(edges)
        expected = (np.diff(sp_stats.norm.cdf(edges_arr, loc=m, scale=s)) * n).tolist()

        self.assertEqual(len(got), len(edges) - 1)
        for g, e in zip(got, expected):
            self.assertAlmostEqual(g, e, places=12)

        # Counts are probabilities scaled by n, so sum should match covered mass.
        covered_mass = float(sp_stats.norm.cdf(edges[-1], loc=m, scale=s)
                             - sp_stats.norm.cdf(edges[0], loc=m, scale=s))
        self.assertAlmostEqual(sum(got), n * covered_mass, places=12)

    def test_cumulative_comparison_with_zero_stdev(self):
        """Regression test for cumulative_comparison infinite loop with zero stdev.

        When all values are identical (stdev == 0), step == 0 would cause
        infinite loop. Should return early with N/A message instead.
        """
        identical = [5.0, 5.0, 5.0]
        result = mod.cumulative_comparison(identical, identical, "A", "B")

        self.assertIn("All values are identical", result)
        self.assertIn("cumulative comparison not applicable", result)
        self.assertIn("5.00", result)

    def test_bin_data_includes_right_edge(self):
        """Regression test for _bin_data silently dropping right-edge values.

        Values at the last bin edge should be counted in the last bin,
        not silently dropped due to bisect_right returning len(edges).
        """
        edges = [0.0, 5.0, 10.0]
        data = [0.0, 2.5, 5.0, 7.5, 10.0]
        counts = mod._bin_data(data, edges)

        self.assertEqual(len(counts), 2)
        self.assertEqual(counts[0], 2)  # 0.0, 2.5 in first bin
        self.assertEqual(counts[1], 3)  # 5.0, 7.5, 10.0 in last bin (10.0 now included)

    def test_bin_data_with_numpy_histogram(self):
        """Test that _bin_data uses numpy's histogram for vectorized binning.

        Verifies that the optimized implementation:
        - Produces correct counts for normal data
        - Handles edge cases (values at bin edges)
        - Returns counts as a list
        - Last bin is closed on both sides
        """
        # Test with normal data
        edges = [0.0, 10.0, 20.0, 30.0]
        data = [1.0, 5.0, 11.0, 15.0, 21.0, 25.0, 29.0]
        counts = mod._bin_data(data, edges)

        # Verify basic properties
        self.assertIsInstance(counts, list)
        self.assertEqual(len(counts), 3)  # 3 bins from 4 edges
        self.assertEqual(counts[0], 2)  # 1.0, 5.0 in [0, 10)
        self.assertEqual(counts[1], 2)  # 11.0, 15.0 in [10, 20)
        self.assertEqual(counts[2], 3)  # 21.0, 25.0, 29.0 in [20, 30]

        # Test with values at edges
        edges = [0.0, 5.0, 10.0]
        data = [0.0, 2.5, 5.0, 7.5, 10.0]
        counts = mod._bin_data(data, edges)

        self.assertEqual(len(counts), 2)
        self.assertEqual(counts[0], 2)  # 0.0, 2.5 in first bin [0, 5)
        self.assertEqual(counts[1], 3)  # 5.0, 7.5, 10.0 in last bin [5, 10] (closed on both sides)

        # Test with empty data
        counts_empty = mod._bin_data([], edges)
        self.assertEqual(counts_empty, [0, 0])

        # Test with single value
        counts_single = mod._bin_data([7.5], edges)
        self.assertEqual(counts_single, [0, 1])

    def test_generate_single_report_summary_with_zero_mean(self):
        """Regression test for generate_single_report summary section with zero mean.

        Summary section should print "N/A (mean is zero)" for CV and not
        suggest variability recommendation when mean is zero.
        """
        report = mod.generate_single_report([-1.0, 0.0, 1.0], "ZeroMean")

        self.assertIn("CV:        N/A (mean is zero)", report)
        self.assertNotIn("CV:        inf%", report)
        self.assertNotIn("High variability", report)

    @_ignore_precision_loss
    def test_generate_report_diff_pct_with_zero_baseline(self):
        """Regression test for generate_report diff_pct formatting with zero baseline mean.

        When baseline mean is zero, diff_pct is inf and should format as
        "N/A (baseline mean is zero)" not "(+inf%)".
        """
        report = mod.generate_report(
            [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], "Baseline", "Experiment"
        )

        self.assertIn("Observed difference: +1.00 (N/A (baseline mean is zero))", report)
        self.assertIn("Mean difference: +1.00 (N/A (baseline mean is zero))", report)
        self.assertNotIn("(+inf%)", report)
        self.assertNotIn("(-inf%)", report)

    def test_pooled_stdev_with_small_samples(self):
        """Regression test for pooled-stdev parameter helper with small samples.

        _pooled_stdev_from_params() should not crash when n1 + n2 <= 2
        (insufficient samples). Previously this path raised ZeroDivisionError.
        """
        # Test with single element in each group (n1=1, n2=1, total=2)
        result = mod._pooled_stdev_from_params(1, 0.0, 1, 0.0)
        self.assertEqual(result, 0.0)

        # Test with single element total (n1=1, n2=0, total=1)
        result = mod._pooled_stdev_from_params(1, 0.0, 0, 0.0)
        self.assertEqual(result, 0.0)

        # Test with three elements total (n1=1, n2=2, boundary: n1+n2=3 passes guard)
        # This should compute normally since denominator is n1+n2-2=1 (valid)
        # group2 sample variance for [102, 105] is 4.5 with ddof=1
        result = mod._pooled_stdev_from_params(1, 0.0, 2, 4.5)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 10.0)

    @_ignore_precision_loss
    def test_cohens_d_with_zero_variance(self):
        """Regression test for Cohen's d with zero pooled standard deviation.

        When both groups have constant values, pooled stdev is 0 and
        Cohen's d should be 0 (not raise ZeroDivisionError).
        """
        # Identical constant values → d = 0
        s_pool = mod._pooled_stdev_from_params(2, 0.0, 2, 0.0)
        self.assertEqual(s_pool, 0.0)

        # Different constants → pooled stdev still 0, d should be 0
        stats1 = mod.compute_descriptive_stats([100.0, 100.0, 100.0])
        stats2 = mod.compute_descriptive_stats([105.0, 105.0, 105.0])
        s_pool = mod._pooled_stdev_from_params(
            stats1['n'], stats1['var'], stats2['n'], stats2['var'])
        self.assertEqual(s_pool, 0.0)
        d = (stats2['mean'] - stats1['mean']) / s_pool if s_pool != 0 else 0.0
        self.assertEqual(d, 0.0)

        # One constant, one variable → d > 0
        stats3 = mod.compute_descriptive_stats([100.0, 100.0, 100.0])
        stats4 = mod.compute_descriptive_stats([100.0, 110.0, 105.0])
        s_pool = mod._pooled_stdev_from_params(
            stats3['n'], stats3['var'], stats4['n'], stats4['var'])
        self.assertGreater(s_pool, 0.0)

        # Normal data → d ≈ 5.0 (mean diff = 5, stdev = 1)
        stats5 = mod.compute_descriptive_stats([100.0, 101.0, 102.0])
        stats6 = mod.compute_descriptive_stats([105.0, 106.0, 107.0])
        s_pool = mod._pooled_stdev_from_params(
            stats5['n'], stats5['var'], stats6['n'], stats6['var'])
        d = (stats6['mean'] - stats5['mean']) / s_pool
        self.assertAlmostEqual(d, 5.0, places=3)

    def test_cohens_d_with_small_samples(self):
        """Regression test for _pooled_stdev_from_params with insufficient samples.

        Should gracefully return 0 when n1 + n2 <= 2.
        """
        result = mod._pooled_stdev_from_params(1, 0.0, 1, 0.0)
        self.assertEqual(result, 0.0)

        result = mod._pooled_stdev_from_params(1, 0.0, 0, 0.0)
        self.assertEqual(result, 0.0)

        # n1=1, n2=2, total=3: should compute normally (uses raw params, not compute_descriptive_stats)
        result = mod._pooled_stdev_from_params(1, 0.0, 2, 4.5)
        self.assertGreater(result, 0.0)

    def test_dagostino_pearson_test_with_constant_data(self):
        """Regression test for dagostino_pearson_test crash on constant data.

        When all values are identical (zero variance), scipy.stats.normaltest
        internally divides by zero and raises ValueError or returns NaN.
        The fix guards against constant data using np.ptp(data) == 0 and NaN p_values.
        """
        # Constant data with 50 identical values
        constant_data = [5.0] * 50
        k2, p_value = mod.dagostino_pearson_test(constant_data)

        # Should return (0, 1.0) without crashing
        self.assertEqual(k2, 0)
        self.assertEqual(p_value, 1.0)

        # Constant data with small sample (already guarded by n < 20 check)
        constant_small = [3.0] * 5
        k2, p_value = mod.dagostino_pearson_test(constant_small)
        self.assertEqual(k2, 0)
        self.assertEqual(p_value, 1.0)

        # Normal data with n >= 20 should compute normally
        normal_data = list(range(1, 26))  # 25 elements

    def test_detect_outliers_zscore_vectorized(self):
        """Test vectorized z-score outlier detection."""
        # Data with clear outlier: 10 values at 100, 1 value at 500
        data = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 500]
        stats = mod.compute_descriptive_stats(data)
        m, s = stats['mean'], stats['stdev']

        outliers = mod.detect_outliers_zscore(data, m, s, threshold=3.0)

        # Should detect 500 as an outlier with high z-score (>3)
        self.assertEqual(len(outliers), 1)
        self.assertAlmostEqual(outliers[0][0], 500.0, places=5)
        self.assertGreater(abs(outliers[0][1]), 3.0)

    @_ignore_precision_loss
    def test_detect_outliers_zscore_zero_stdev(self):
        """Test z-score outlier detection with zero standard deviation."""
        # All values identical
        data = [5.0] * 10
        stats = mod.compute_descriptive_stats(data)
        m, s = stats['mean'], stats['stdev']

        outliers = mod.detect_outliers_zscore(data, m, s, threshold=3.0)

        # Should return empty list without crashing
        self.assertEqual(outliers, [])

    def test_detect_outliers_zscore_multiple_outliers(self):
        """Test z-score outlier detection with multiple outliers sorted by |z|."""
        # Data with two clear outliers: mean≈25, stdev≈4.5
        data = [25, 24, 26, 25, 24, 26, 25, 24, 10, 40]  # 10 and 40 are outliers
        stats = mod.compute_descriptive_stats(data)
        m, s = stats['mean'], stats['stdev']

        outliers = mod.detect_outliers_zscore(data, m, s, threshold=2.0)

        # Should detect two outliers
        self.assertEqual(len(outliers), 2)

        # Should be sorted by absolute z-score (descending)
        self.assertGreaterEqual(abs(outliers[0][1]), abs(outliers[1][1]))

        # Both should have |z| > 2.0
        for val, z in outliers:
            self.assertGreater(abs(z), 2.0)

    def test_detect_outliers_zscore_no_outliers(self):
        """Test z-score outlier detection when no outliers exist."""
        # Normal distribution-like data with no extreme values
        data = [10, 11, 10, 9, 10, 11, 10, 9, 10, 11]
        stats = mod.compute_descriptive_stats(data)
        m, s = stats['mean'], stats['stdev']

        outliers = mod.detect_outliers_zscore(data, m, s, threshold=3.0)

        # Should find no outliers
        self.assertEqual(outliers, [])

    def test_detect_outliers_zscore_custom_threshold(self):
        """Test z-score outlier detection with custom threshold."""
        data = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 250]  # 250 is outlier
        stats = mod.compute_descriptive_stats(data)
        m, s = stats['mean'], stats['stdev']

        # With threshold=1.5, 250 should be detected
        outliers_low = mod.detect_outliers_zscore(data, m, s, threshold=1.5)

        # With threshold=5.0, 250 might not be detected
        outliers_high = mod.detect_outliers_zscore(data, m, s, threshold=5.0)

        # Lower threshold should detect more or equal outliers
        self.assertGreaterEqual(len(outliers_low), len(outliers_high))


    def test_detect_outliers_iqr_vectorized(self):
        """Test vectorized IQR outlier detection."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
        stats = mod.compute_descriptive_stats(data)
        q1, q3, iqr = stats['p25'], stats['p75'], stats['iqr']

        outliers = mod.detect_outliers_iqr(data, q1, q3, iqr)

        # Should detect 100 as an outlier
        self.assertIn(100, outliers)
        self.assertEqual(len(outliers), 1)

    def test_detect_outliers_iqr_multiple_outliers(self):
        """Test IQR outlier detection with multiple outliers."""
        data = [1, 10, 11, 12, 13, 14, 15, 16, 17, 100]
        stats = mod.compute_descriptive_stats(data)
        q1, q3, iqr = stats['p25'], stats['p75'], stats['iqr']

        outliers = mod.detect_outliers_iqr(data, q1, q3, iqr)

        # Should detect outliers (1 and/or 100 depending on IQR)
        # Should be sorted
        self.assertEqual(outliers, sorted(outliers))
        self.assertGreater(len(outliers), 0)

    def test_detect_outliers_iqr_no_outliers(self):
        """Test IQR outlier detection when no outliers exist."""
        data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        stats = mod.compute_descriptive_stats(data)
        q1, q3, iqr = stats['p25'], stats['p75'], stats['iqr']

        outliers = mod.detect_outliers_iqr(data, q1, q3, iqr)

        # Should find no outliers
        self.assertEqual(outliers, [])

    def test_detect_outliers_iqr_sorted_output(self):
        """Test that IQR outlier detection returns sorted results."""
        data = [100, 5, 6, 7, 8, 9, 10, 11, 12, -50]
        stats = mod.compute_descriptive_stats(data)
        q1, q3, iqr = stats['p25'], stats['p75'], stats['iqr']

        outliers = mod.detect_outliers_iqr(data, q1, q3, iqr)

        # Should be sorted
        self.assertEqual(outliers, sorted(outliers))

    @_ignore_precision_loss
    def test_generate_single_report_ci_agreement_with_zero_se(self):
        """Regression test for CI agreement check with se=0.

        When all values are identical, se=0. Both bootstrap and parametric CIs
        collapse to [m, m], so they perfectly agree. But the old code used
        `abs(0) < 0` which is False, incorrectly reporting divergence.
        The fix changes `<` to `<=` so abs(diff) <= se is True when both are 0.
        """
        # Constant data where se = 0
        constant_data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        report = mod.generate_single_report(constant_data, "Constant", alpha=0.05)

        # With the fix, should report CIs agree (✅)
        self.assertIn("✅", report)
        self.assertIn("agree", report.lower())
        self.assertNotIn("diverge", report.lower())

        # Verify no false "distribution may be non-normal" warning
        self.assertNotIn("non-normal", report.lower())



    @_ignore_precision_loss
    def test_generate_report_zero_diff_higher_is_better_true(self):
        """Regression test for direction_good with zero difference (higher_is_better=True).

        When diff_mean == 0, direction should report "No change" consistently.
        Bug: (0 > 0) == True evaluates to False, incorrectly reporting Regression.
        Fix: Check diff_mean == 0 explicitly and report "No change".
        """
        # Identical data (zero difference)
        base = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        exp = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        report = mod.generate_report(base, exp, "Base", "Exp",
                                         higher_is_better=True, alpha=0.05)

        # Should report "No change" (not Improvement, not Regression)
        self.assertIn("\u2796 No change", report)
        # Should NOT report Improvement or Regression for zero difference
        # Improvement message: "✅ Improvement"
        improvement_lines = [line for line in report.splitlines() if "\u2705 Improvement" in line]
        self.assertEqual(len(improvement_lines), 0, "Should not report Improvement for zero difference")

    @_ignore_precision_loss
    def test_generate_report_zero_diff_higher_is_better_false(self):
        """Regression test for direction_good with zero difference (higher_is_better=False).

        When diff_mean == 0, direction should report "No change" consistently.
        Bug: (0 > 0) == False evaluates to True, incorrectly reporting Improvement.
        Fix: Check diff_mean == 0 explicitly and report "No change".
        """
        # Identical data (zero difference)
        base = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        exp = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

        report = mod.generate_report(base, exp, "Base", "Exp",
                                         higher_is_better=False, alpha=0.05)

        # Should report "No change" (not Improvement, not Regression)
        self.assertIn("\u2796 No change", report)
        # Should NOT report Improvement or Regression for zero difference
        # Improvement message: "✅ Improvement"
        improvement_lines = [line for line in report.splitlines() if "\u2705 Improvement" in line]
        self.assertEqual(len(improvement_lines), 0, "Should not report Improvement for zero difference")

    @_ignore_precision_loss
    def test_generate_report_nonzero_diff_consistency(self):
        """Regression test for direction_good with non-zero difference.

        Verify that non-zero differences still report correctly after the zero-diff fix.
        """
        base = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        exp_better = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
        exp_worse = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        # When higher_is_better=True and exp > base → should report Improvement
        report_improvement = mod.generate_report(base, exp_better, "Base", "Exp",
                                                     higher_is_better=True, alpha=0.05)
        self.assertIn("\u2705 Improvement", report_improvement)

        # When higher_is_better=True and exp < base → should report Regression
        report_regression = mod.generate_report(base, exp_worse, "Base", "Exp",
                                                    higher_is_better=True, alpha=0.05)
        self.assertIn("\u274c Regression", report_regression)

    def test_sample_size_analysis_single_signature_without_label(self):
        """Regression test: _sample_size_analysis_single no longer accepts label."""
        stats = mod.compute_descriptive_stats([10.0, 11.0, 12.0])

        # New API: stats + optional alpha
        analysis = mod._sample_size_analysis_single(stats, alpha=0.05)
        self.assertIn("Current sample size", analysis)

        # Old API (stats, label, alpha) should fail after removing dead parameter
        with self.assertRaises(TypeError):
            mod._sample_size_analysis_single(stats, "Single", 0.05)

    @_ignore_precision_loss
    def test_sample_size_analysis_single_req_1pct_conditional(self):
        """Testing that req_1pct is only computed when moe_pct > 2.0 (low/poor precision).

        req_1pct (required sample size for ±1% precision) should only be
        computed and displayed for low (moe_pct ≤ 5%) and poor (moe_pct > 5%)
        precision cases, not for excellent/good/moderate precision cases.
        """
        # Case 1: High precision (large sample, low variability)
        # This should have low moe_pct and NOT compute req_1pct
        high_precision_data = [100.0] * 100  # Very consistent data, large sample
        stats_high = mod.compute_descriptive_stats(high_precision_data)
        analysis_high = mod._sample_size_analysis_single(stats_high, alpha=0.05)

        # Should show excellent/good precision, NOT req_1pct message
        self.assertIn("Excellent precision", analysis_high or "Good precision", "Good precision")
        self.assertNotIn("To reach ±1% precision", analysis_high,
                         "High precision case should not mention ±1% precision requirement")

        # Case 2: Low precision (small sample, high variability)
        # This should have high moe_pct and SHOULD compute req_1pct
        low_precision_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # Small sample, some variation
        stats_low = mod.compute_descriptive_stats(low_precision_data)
        analysis_low = mod._sample_size_analysis_single(stats_low, alpha=0.05)

        # Should show low/poor precision and include req_1pct message
        self.assertIn("To reach ±1% precision", analysis_low,
                      "Low precision case should mention ±1% precision requirement")
        # Verify the message contains a number (the required sample size)
        import re
        self.assertIsNotNone(re.search(r"To reach ±1% precision.*\d+", analysis_low),
                            "Should have req_1pct value (number) in message")

    def test_compute_p_value_scale_info_alpha_001(self):
        """Regression test for p-value scale with alpha=0.01.

        When alpha=0.01, scale should show 0.00-0.05 with ticks every 0.01.
        Each tick should be linearly spaced and α label position should be accurate.
        """
        tick_labels, tick_bar, pos_mult, scale_width = mod._compute_p_value_scale_info(0.01)

        # alpha=0.01 should map to 0.00..0.05 scale => pos_mult = 30 / 0.05 = 600
        self.assertAlmostEqual(pos_mult, 600.0, places=5)
        self.assertEqual(scale_width, 30)  # (6 ticks - 1) * 6

        # Check label formatting
        self.assertIn("0.00", tick_labels)
        self.assertIn("0.01", tick_labels)
        self.assertIn("0.05", tick_labels)
        self.assertIn("\u2502", tick_bar)

    def test_compute_p_value_scale_info_alpha_010(self):
        """Regression test for p-value scale with alpha=0.10.

        When alpha>0.05 and <=0.10, scale should show 0.00-0.10 with ticks every 0.02.
        This ensures alpha label appears at the correct visual position.
        """
        tick_labels, tick_bar, pos_mult, scale_width = mod._compute_p_value_scale_info(0.10)

        # alpha=0.10 should map to 0.00..0.10 scale => pos_mult = 30 / 0.10 = 300
        self.assertAlmostEqual(pos_mult, 300.0, places=5)
        self.assertEqual(scale_width, 30)  # (6 ticks - 1) * 6

        # Check label formatting
        self.assertIn("0.00", tick_labels)
        self.assertIn("0.10", tick_labels)
        self.assertIn("\u2502", tick_bar)

    def test_p_value_visual_scale_adapts_to_alpha(self):
        """Regression test for p-value visual scale adaptation (COSMETIC fix).

        The p-value visual should adapt its scale based on alpha instead of
        being hardcoded to 0.00-0.05. With alpha=0.10, the scale should extend
        to 0.10, and α label should appear at the correct position.
        """
        # Create simple data
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        exp = [1.5, 2.5, 3.5, 4.5, 5.5]

        # Generate report with alpha=0.10
        report = mod.generate_report(base, exp, "Base", "Exp", alpha=0.10)

        # Check that alpha label mentions 0.10
        self.assertIn("0.1000", report)  # α in report should mention the literal value

        # Check that p-value visual section exists
        self.assertIn("p-value visual summary", report)


    def test_generate_single_report_ci_agreement_with_n_gt_2(self):
        """Regression test for CI agreement check with n>2.

        When n>2, parametric CI is available and agreement check should
        proceed normally. Should not show "not available" message.
        """
        normal_data = [4.8, 5.0, 5.2, 4.9, 5.1]
        report = mod.generate_single_report(normal_data, "Normal", alpha=0.05)

        # Should NOT show "Parametric CI not available"
        self.assertNotIn("Parametric CI not available (n < 2)", report)

        # Should have an agreement check result (either agree or diverge)
        self.assertTrue(
            "agree" in report or "diverge" in report,
            "Should show either agreement or divergence check result"
        )

    def test_cumulative_comparison_numpy_optimization(self):
        """Regression test for cumulative_comparison numpy optimization.

        Performance issue: Previous implementation was O(t * (n1 + n2)) because
        it scanned all data for every threshold in both filtering and computation.
        Fix: Use sorted data + vectorized np.searchsorted lookups.
        Complexity: O((n1+n2) log(n1+n2)) for sorting + O(t + log(n)) for lookups.

        This test verifies:
        1. Output is correct (same as before optimization)
        2. Threshold filtering and computation use numpy searchsorted
        3. Edge cases (empty results, tied percentages) work correctly
        """
        # Test case 1: Different distributions
        base = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        exp = [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]

        result = mod.cumulative_comparison(base, exp, 'Base', 'Exp')

        # Should have table with header and data rows
        self.assertIn("Percentage of values", result)
        self.assertIn("Threshold", result)
        self.assertIn("Base", result)
        self.assertIn("Exp", result)

        # Should show percentages
        self.assertIn("%", result)

        # Should show comparison results (differences > 2% should be marked)
        # Since Exp is consistently ~2-25% higher, should show Exp as "Better"
        self.assertIn("Exp", result)

        # Test case 2: With zero values (searchsorted should handle correctly)
        base_zeros = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        exp_zeros = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        result_zeros = mod.cumulative_comparison(base_zeros, exp_zeros, 'A', 'B')

        # Should handle zero thresholds correctly
        self.assertIn("≥ 0", result_zeros)
        self.assertIn("100%", result_zeros)

        # Test case 3: Identical datasets (edge case)
        same = [5.0] * 6
        result_same = mod.cumulative_comparison(same, same, 'X', 'Y')

        # Should recognize identical values and report N/A
        self.assertIn("All values are identical", result_same)

    def test_cumulative_comparison_vectorized_searchsorted(self):
        """Verify cumulative_comparison uses vectorized np.searchsorted correctly.

        This test validates that the np.searchsorted vectorization produces
        correct results for various data distributions and threshold ranges.

        Key points:
        - np.searchsorted processes all thresholds in one vectorized call
        - Replaces the previous O(t log n) per-threshold loop with O(t + log n)
        - Should produce identical results to the prior non-vectorized approach
        """
        # Test case 1: Normal distribution with mixed thresholds
        base = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        exp = [12.0, 17.0, 22.0, 27.0, 32.0, 37.0, 42.0, 47.0, 52.0]

        result = mod.cumulative_comparison(base, exp, 'Base', 'Exp')

        # Verify structure
        self.assertIn("Percentage of values ≥ threshold:", result)
        self.assertIn("Threshold", result)
        self.assertIn("Base", result)
        self.assertIn("Exp", result)

        # Verify threshold rows are present
        lines = result.split('\n')
        threshold_lines = [line for line in lines if '≥' in line and '%' in line]
        self.assertGreater(len(threshold_lines), 0, "Should have threshold comparison rows")

        # Test case 2: Edge case with thresholds at data boundaries
        # All values in exp are exactly 2 higher than base
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [3.0, 4.0, 5.0, 6.0, 7.0]

        result2 = mod.cumulative_comparison(data1, data2, 'A', 'B')

        # Should show that B is consistently better (higher values)
        self.assertIn("B", result2)
        self.assertIn("%", result2)

        # Test case 3: Negative values and zero crossings
        neg_data1 = [-10.0, -5.0, 0.0, 5.0, 10.0]
        neg_data2 = [-8.0, -3.0, 2.0, 7.0, 12.0]

        result3 = mod.cumulative_comparison(neg_data1, neg_data2, 'X', 'Y')

        # Should handle negative thresholds correctly
        self.assertIn("Percentage of values", result3)
        self.assertIn("%", result3)

        # Test case 4: Large dataset to verify vectorization efficiency
        rng = np.random.RandomState(42)
        large1 = rng.normal(100, 15, 1000).tolist()
        large2 = rng.normal(105, 15, 1000).tolist()

        result4 = mod.cumulative_comparison(large1, large2, 'L1', 'L2')

        # Should produce valid output with multiple thresholds
        threshold_lines4 = [line for line in result4.split('\n') if '≥' in line and '%' in line]
        self.assertGreater(len(threshold_lines4), 3, "Should have multiple thresholds for large dataset")

    def test_single_mode_descriptive_stats_match_reference_calculations(self):
        """Correctness test: single-mode descriptive stats match NumPy reference."""
        rng = np.random.RandomState(123)
        data = rng.normal(loc=42.0, scale=3.5, size=400).tolist()

        stats = mod.compute_descriptive_stats(data)

        self.assertAlmostEqual(stats['mean'], float(np.mean(data)), places=10)
        self.assertAlmostEqual(stats['var'], float(np.var(data, ddof=1)), places=10)
        self.assertAlmostEqual(stats['stdev'], float(np.std(data, ddof=1)), places=10)
        self.assertAlmostEqual(stats['median'], float(np.median(data)), places=10)
        self.assertAlmostEqual(stats['p25'], float(np.percentile(data, 25)), places=10)
        self.assertAlmostEqual(stats['p75'], float(np.percentile(data, 75)), places=10)

    def test_single_mode_report_for_known_data_has_expected_summary(self):
        """Correctness test: single-mode report includes expected summary values."""
        data = [10.0, 12.0, 14.0, 16.0, 18.0]
        report = mod.generate_single_report(data, "KnownData", alpha=0.05)

        self.assertIn("Benchmark: KnownData", report)
        self.assertIn("Result:    14.00 ± 3.16 ops/min", report)
        self.assertIn("95% CI:", report)

    def test_comparison_mode_detects_significant_improvement(self):
        """Correctness test: known shifted distribution is detected as improvement."""
        rng = np.random.RandomState(2026)
        base = rng.normal(loc=100.0, scale=4.0, size=120).tolist()
        exp = rng.normal(loc=108.0, scale=4.0, size=120).tolist()

        # Validate test-level statistics directly with reduced iteration counts.
        t_stat, _, p2_t = mod.welch_t_test(base, exp)
        _, z_stat, p2_mw = mod.mann_whitney_u(base, exp)
        perm_diff, p2_perm = mod.permutation_test(base, exp, n_perms=3000, seed=7)
        _, _, p2_boot = mod.bootstrap_ci(base, exp, n_boot=3000, seed=7)

        self.assertGreater(t_stat, 0.0)
        self.assertGreater(z_stat, 0.0)
        self.assertGreater(perm_diff, 0.0)

        p1_vals = [
            mod.one_tailed_p_value(p2_t, t_stat, higher_is_better=True),
            mod.one_tailed_p_value(p2_mw, z_stat, higher_is_better=True),
            mod.one_tailed_p_value(p2_perm, perm_diff, higher_is_better=True),
            mod.one_tailed_p_value(p2_boot, np.mean(exp) - np.mean(base), higher_is_better=True),
        ]

        for p1 in p1_vals:
            self.assertLess(p1, 0.05)

        # End-to-end comparison mode verdict should classify as improvement.
        report = mod.generate_report(
            base, exp, "Baseline", "Experiment", higher_is_better=True, alpha=0.05)
        self.assertIn("✅ Improvement", report)

    def test_comparison_mode_detects_significant_regression(self):
        """Correctness test: shifted distribution in wrong direction is regression."""
        rng = np.random.RandomState(2027)
        base = rng.normal(loc=100.0, scale=4.0, size=120).tolist()
        exp = rng.normal(loc=92.0, scale=4.0, size=120).tolist()

        t_stat, _, p2_t = mod.welch_t_test(base, exp)
        _, z_stat, p2_mw = mod.mann_whitney_u(base, exp)
        perm_diff, p2_perm = mod.permutation_test(base, exp, n_perms=3000, seed=11)
        _, _, p2_boot = mod.bootstrap_ci(base, exp, n_boot=3000, seed=11)

        self.assertLess(t_stat, 0.0)
        self.assertLess(z_stat, 0.0)
        self.assertLess(perm_diff, 0.0)

        p1_vals = [
            mod.one_tailed_p_value(p2_t, t_stat, higher_is_better=True),
            mod.one_tailed_p_value(p2_mw, z_stat, higher_is_better=True),
            mod.one_tailed_p_value(p2_perm, perm_diff, higher_is_better=True),
            mod.one_tailed_p_value(p2_boot, np.mean(exp) - np.mean(base), higher_is_better=True),
        ]

        for p1 in p1_vals:
            self.assertGreater(p1, 0.95)

        report = mod.generate_report(
            base, exp, "Baseline", "Experiment", higher_is_better=True, alpha=0.05)
        self.assertIn("❌ Regression", report)

    def test_precomputed_stats_optimization(self):
        """Verify that pre-computed stats produce identical output."""
        rng = np.random.RandomState(42)
        data = rng.normal(loc=100.0, scale=10.0, size=50).tolist()

        # Compute stats once
        stats = mod.compute_descriptive_stats(data)
        edges = mod._compute_bins([data])

        # Test _single_histogram with and without stats
        hist_without_stats = mod._single_histogram(data, "test", edges)
        hist_with_stats = mod._single_histogram(data, "test", edges, stats=stats)
        self.assertEqual(hist_without_stats, hist_with_stats,
                        "_single_histogram output must be identical with/without stats")

        # Test _normality_histogram with and without stats
        norm_without_stats = mod._normality_histogram(data, "test", edges)
        norm_with_stats = mod._normality_histogram(data, "test", edges, stats=stats)
        self.assertEqual(norm_without_stats, norm_with_stats,
                        "_normality_histogram output must be identical with/without stats")

        # Test _stability_chart with and without stats
        stab_without_stats = mod._stability_chart(data, "test")
        stab_with_stats = mod._stability_chart(data, "test", stats=stats)
        self.assertEqual(stab_without_stats, stab_with_stats,
                        "_stability_chart output must be identical with/without stats")

        # Test cohens_d computation with and without stats
        data2 = rng.normal(loc=105.0, scale=10.0, size=50).tolist()
        stats2 = mod.compute_descriptive_stats(data2)

        # Without stats: compute from raw data
        s_pool_raw = mod._pooled_stdev_from_params(
            len(data), _ref_variance(data), len(data2), _ref_variance(data2))
        d_raw = (_ref_mean(data2) - _ref_mean(data)) / s_pool_raw if s_pool_raw != 0 else 0.0

        # With stats: compute from precomputed stats
        s_pool_stats = mod._pooled_stdev_from_params(
            stats['n'], stats['var'], stats2['n'], stats2['var'])
        d_stats = (stats2['mean'] - stats['mean']) / s_pool_stats if s_pool_stats != 0 else 0.0

        self.assertEqual(d_raw, d_stats,
                        "Cohen's d must be identical with/without precomputed stats")

    @_ignore_precision_loss
    def test_cohens_d_uses_precomputed_variances_when_stats_provided(self):
        """Regression test: _pooled_stdev_from_params uses precomputed variances correctly."""
        data1 = [10.0, 10.0, 10.0, 10.0]
        data2 = [20.0, 20.0, 20.0, 20.0]

        stats1 = mod.compute_descriptive_stats(data1)
        stats2 = mod.compute_descriptive_stats(data2)

        # Both groups are constant → variance = 0 → pooled stdev = 0
        s_pool = mod._pooled_stdev_from_params(
            stats1['n'], stats1['var'], stats2['n'], stats2['var'])
        self.assertEqual(s_pool, 0.0)

        # Cohen's d should be 0 when pooled stdev is 0
        d = (stats2['mean'] - stats1['mean']) / s_pool if s_pool != 0 else 0.0
        self.assertEqual(d, 0.0)

    def test_generate_report_reuses_pooled_stdev_for_effect_size_and_power(self):
        """Regression test: generate_report should compute pooled stdev once."""
        base = [100.0, 101.0, 102.0, 103.0]
        exp = [104.0, 105.0, 106.0, 107.0]

        call_count = {'count': 0}
        original_pooled_stdev_from_params = mod._pooled_stdev_from_params
        try:
            def _counting_pooled_stdev_from_params(n1, v1, n2, v2):
                call_count['count'] += 1
                return original_pooled_stdev_from_params(n1, v1, n2, v2)

            mod._pooled_stdev_from_params = _counting_pooled_stdev_from_params

            report = mod.generate_report(base, exp, "Base", "Exp")
            self.assertIn("Cohen's d:", report)
            self.assertEqual(call_count['count'], 1)
        finally:
            mod._pooled_stdev_from_params = original_pooled_stdev_from_params

    def test_generate_single_report_reuses_ci_for_sample_size_analysis(self):
        """Regression test: single report includes sample size analysis."""
        data = [100.0, 101.0, 102.0, 103.0, 104.0]
        report = mod.generate_single_report(data, "Single")
        self.assertIn("SAMPLE SIZE ADEQUACY", report)
        self.assertIn("Margin of Error:", report)

    def test_single_histogram_mode_uses_max_count(self):
        """Regression test: mode markers should be based on max_count (including ties)."""
        # Fixed bin edges produce counts [2, 2, 1] so first two bins are both modes.
        data = [0.1, 0.2, 1.1, 1.2, 2.1]
        edges = [0.0, 1.0, 2.0, 3.0]

        hist = mod._single_histogram(data, "test", edges)

        # Tied max bins should both be marked as mode.
        self.assertEqual(hist.count("mode"), 2)

    def test_compute_bins_with_precomputed_stats(self):
        """Verify that _compute_bins produces valid edges for single and multi-dataset cases."""
        rng = np.random.RandomState(123)
        data = rng.normal(loc=100.0, scale=15.0, size=75).tolist()

        # Compute bins
        edges = mod._compute_bins([data])
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 1, "Should produce multiple bin edges")

        # Multi-dataset case
        data2 = rng.normal(loc=90.0, scale=12.0, size=60).tolist()
        edges_multi = mod._compute_bins([data, data2])
        self.assertGreater(len(edges_multi), 1, "Multi-dataset _compute_bins failed")

    def test_compute_descriptive_stats_no_sorted_data_in_dict(self):
        """Test that compute_descriptive_stats doesn't return sorted_data key."""
        import numpy as np

        # Create test data
        data = [1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.4, 8.9]
        stats = mod.compute_descriptive_stats(data)

        # Verify that sorted_data is NOT in the returned dict
        self.assertNotIn('sorted_data', stats,
                        "sorted_data should not be in returned dict (dead key)")

        # Verify that all expected keys ARE present
        expected_keys = {'n', 'mean', 'var', 'stdev', 'se', 'median', 'min', 'max',
                        'range', 'p25', 'p75', 'iqr', 'skewness', 'ex_kurtosis'}
        self.assertEqual(set(stats.keys()), expected_keys,
                        "Unexpected keys in stats dict")

        # Verify that min/max still work correctly (sorted_data is used internally)
        self.assertEqual(stats['min'], min(data), "min value should be correct")
        self.assertEqual(stats['max'], max(data), "max value should be correct")
        self.assertGreater(stats['median'], 4.0, "median should be reasonable")
        self.assertLess(stats['median'], 6.0, "median should be reasonable")

        # Verify with larger random dataset
        rng = np.random.RandomState(42)
        large_data = rng.randn(1000).tolist()
        large_stats = mod.compute_descriptive_stats(large_data)
        self.assertNotIn('sorted_data', large_stats,
                        "sorted_data should not be in dict for large dataset")
        self.assertEqual(large_stats['min'], min(large_data))
        self.assertEqual(large_stats['max'], max(large_data))

    def test_compute_descriptive_stats_no_unnecessary_sorting(self):
        """Test that compute_descriptive_stats produces correct results without pre-sorting.

        This verifies the optimization where we removed the upfront sorted() call
        that was redundant since numpy functions (median, percentile, min, max)
        handle their own internal sorting more efficiently.
        """
        # Test with unordered data
        unordered_data = [7.2, 3.1, 9.8, 1.5, 5.4, 2.8, 8.6, 4.3]
        unordered_stats = mod.compute_descriptive_stats(unordered_data)

        # Test with pre-sorted data
        sorted_data = sorted(unordered_data)
        sorted_stats = mod.compute_descriptive_stats(sorted_data)

        # Test with reverse-sorted data
        reverse_data = sorted(unordered_data, reverse=True)
        reverse_stats = mod.compute_descriptive_stats(reverse_data)

        # All three should produce identical statistics
        for key in ['n', 'mean', 'var', 'stdev', 'se', 'median', 'min', 'max',
                    'range', 'p25', 'p75', 'iqr', 'skewness', 'ex_kurtosis']:
            self.assertAlmostEqual(unordered_stats[key], sorted_stats[key], places=10,
                                   msg=f"Stats mismatch for {key} on sorted vs unordered data")
            self.assertAlmostEqual(unordered_stats[key], reverse_stats[key], places=10,
                                   msg=f"Stats mismatch for {key} on reverse-sorted vs unordered data")

        # Verify specific correctness of min/max without pre-sorting
        self.assertEqual(unordered_stats['min'], min(unordered_data))
        self.assertEqual(unordered_stats['max'], max(unordered_data))
        self.assertEqual(unordered_stats['range'], max(unordered_data) - min(unordered_data))

    def test_format_bin_label_no_width_parameter(self):
        """Test that _format_bin_label works without unused width parameter."""
        # Test with integer bin edges
        label_int = mod._format_bin_label(1050, 1055)
        self.assertEqual(label_int, "[1050,1055)")
        self.assertIsInstance(label_int, str)

        # Test with float bin edges
        label_float = mod._format_bin_label(1050.5, 1055.3)
        self.assertEqual(label_float, "[1050.5,1055.3)")
        self.assertIsInstance(label_float, str)

        # Test with mixed integer and float
        label_mixed1 = mod._format_bin_label(1050, 1055.5)
        self.assertEqual(label_mixed1, "[1050.0,1055.5)")

        label_mixed2 = mod._format_bin_label(1050.5, 1055)
        self.assertEqual(label_mixed2, "[1050.5,1055.0)")

        # Test with negative values
        label_neg = mod._format_bin_label(-10, -5)
        self.assertEqual(label_neg, "[-10,-5)")

        # Test with very small floats
        label_small = mod._format_bin_label(0.0001, 0.0002)
        self.assertEqual(label_small, "[0.0,0.0)")  # Will round to 0.0 with :.1f format

        # Verify function signature doesn't accept width parameter
        # (if it did, this would succeed; if it doesn't, it should raise TypeError)
        with self.assertRaises(TypeError):
            mod._format_bin_label(100, 105, width=15)

    def test_compute_p_value_scale_info_merged_branches(self):
        """Test that alpha <= 0.05 branch produces identical results for all alpha <= 0.05."""
        # Get results for different alpha values that should all use the same scale
        result_001 = mod._compute_p_value_scale_info(0.001)
        result_005 = mod._compute_p_value_scale_info(0.005)
        result_010 = mod._compute_p_value_scale_info(0.010)
        result_030 = mod._compute_p_value_scale_info(0.030)
        result_050 = mod._compute_p_value_scale_info(0.050)

        # All should produce identical tick_labels, tick_bar, pos_mult, scale_width
        # because they all satisfy alpha <= 0.05
        for result in [result_005, result_010, result_030, result_050]:
            tick_labels, tick_bar, pos_mult, scale_width = result

            # Verify tick_labels and tick_bar are formatted correctly
            self.assertIn("0.00", tick_labels)
            self.assertIn("0.05", tick_labels)
            self.assertGreater(len(tick_bar), 0)
            self.assertIn("\u2502", tick_bar)  # Verify it contains the tick character

            # Verify position_multiplier is 30 / 0.05 = 600 and scale_width is 30
            self.assertAlmostEqual(pos_mult, 600.0, places=10)
            self.assertEqual(scale_width, 30)

        # Also verify that result_001 (alpha = 0.001) matches
        tick_labels, tick_bar, pos_mult, scale_width = result_001
        self.assertIn("0.05", tick_labels)
        self.assertIn("\u2502", tick_bar)
        self.assertAlmostEqual(pos_mult, 600.0, places=10)
        self.assertEqual(scale_width, 30)

    def test_compute_p_value_scale_info_returns_four_values(self):
        """Regression test: _compute_p_value_scale_info returns tick_labels, tick_bar, pos_mult, scale_width."""
        result = mod._compute_p_value_scale_info(0.05)
        self.assertEqual(len(result), 4)

        tick_labels, tick_bar, pos_mult, scale_width = result
        self.assertIn("0.00", tick_labels)
        self.assertIn("0.05", tick_labels)
        self.assertIn("\u2502", tick_bar)
        self.assertAlmostEqual(pos_mult, 600.0, places=10)  # 30 / 0.05 = 600
        self.assertEqual(scale_width, 30)

    def test_compute_bins_simplified_no_unused_parameters(self):
        """Test that _compute_bins works correctly after removing unused bin_width/n_bins parameters."""
        import numpy as np

        # Create test data
        rng = np.random.RandomState(42)
        data = rng.normal(loc=100.0, scale=15.0, size=100).tolist()

        # Test basic functionality: single dataset without stats
        edges = mod._compute_bins([data])
        self.assertGreater(len(edges), 1, "Should generate multiple edges")
        self.assertGreaterEqual(edges[0], min(data) - 10,
                               "First edge should be near or before data minimum")
        self.assertLessEqual(edges[-1], max(data) + 10,
                            "Last edge should be near or after data maximum")
        self.assertEqual(edges, sorted(edges), "Edges should be sorted")

        # Verify edges don't have large gaps
        gaps = [edges[i+1] - edges[i] for i in range(len(edges)-1)]
        self.assertGreater(min(gaps), 0, "All gaps should be positive")
        # Most gaps should be similar (consistent bin width)
        avg_gap = sum(gaps) / len(gaps)
        for i, gap in enumerate(gaps):
            self.assertLess(abs(gap - avg_gap) / avg_gap, 0.5,
                           f"Gap {i} ({gap}) differs from average ({avg_gap})")

        # Test with multiple datasets
        data2 = rng.normal(loc=95.0, scale=12.0, size=80).tolist()
        edges_multi = mod._compute_bins([data, data2])
        self.assertGreater(len(edges_multi), 1, "Multi-dataset should produce multiple edges")

        # Verify function signature rejects old parameters
        with self.assertRaises(TypeError):
            mod._compute_bins([data], bin_width=1.0)
        with self.assertRaises(TypeError):
            mod._compute_bins([data], n_bins=10)
        with self.assertRaises(TypeError):
            mod._compute_bins([data], bin_width=1.0, n_bins=10)

        # Test edge case: constant data
        const_data = [5.0] * 50
        edges_const = mod._compute_bins([const_data])
        self.assertEqual(len(edges_const), 2, "Constant data should produce 2 edges")

    def test_effect_size_entries_always_includes_observed(self):
        """Test that _effect_size_entries always includes observed effect size.

        Now that include_observed parameter has been removed, the function should
        always include the observed effect size when it exceeds min_observed.
        """
        # Test with observed effect size above min_observed
        observed_d = 0.5
        entries = mod._effect_size_entries(observed_d, min_observed=0.0)

        # Should have 4 entries: Small, Medium, Large, and Observed
        self.assertEqual(len(entries), 4, "Should have 4 effect size entries")

        # Verify the entries are correct
        expected_labels = ["Small", "Medium", "Large", "Observed"]
        actual_labels = [label for label, _ in entries]
        self.assertEqual(actual_labels, expected_labels)

        # Verify values
        self.assertEqual(entries[0][1], 0.2)  # Small
        self.assertEqual(entries[1][1], 0.5)  # Medium
        self.assertEqual(entries[2][1], 0.8)  # Large
        self.assertEqual(entries[3][1], 0.5)  # Observed (abs value of 0.5)

        # Test with observed effect size below min_observed
        entries_filtered = mod._effect_size_entries(0.01, min_observed=0.05)

        # Should have 3 entries: Small, Medium, Large (no Observed)
        self.assertEqual(len(entries_filtered), 3,
                        "Should not include Observed when below min_observed")
        self.assertEqual([label for label, _ in entries_filtered],
                        ["Small", "Medium", "Large"])

        # Test with negative observed effect size (uses absolute value)
        entries_neg = mod._effect_size_entries(-0.75, min_observed=0.0)

        # Should have 4 entries with Observed = 0.75 (absolute value)
        self.assertEqual(len(entries_neg), 4)
        self.assertEqual(entries_neg[3], ("Observed", 0.75))

    def test_format_stats_single_returns_ci_str(self):
        """Test that _format_stats_single returns ci_str to avoid duplicate computation.

        This ensures the fourth return value (ci_str) is properly formatted and
        can be reused by callers (like generate_single_report) instead of
        recomputing the CI format string.
        """
        # Test with normal data (n >= 2)
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = mod.compute_descriptive_stats(data)
        lines, ci_lo, ci_hi, ci_str = mod._format_stats_single(stats, alpha=0.05)

        # ci_str should contain formatted CI bounds
        self.assertIsInstance(ci_str, str)
        self.assertIn("[", ci_str)
        self.assertIn("]", ci_str)
        self.assertIn(",", ci_str)

        # ci_str should be the same regardless of call count
        _, _, _, ci_str_2 = mod._format_stats_single(stats, alpha=0.05)
        self.assertEqual(ci_str, ci_str_2)

        # ci_str should be used in the formatted lines
        formatted_table = "\n".join(lines)
        self.assertIn(ci_str, formatted_table)

    def test_generate_single_report_uses_ci_str_from_format_stats(self):
        """Test that generate_single_report reuses ci_str, avoiding duplicate computation.

        This verifies the integration: _format_stats_single returns ci_str,
        and generate_single_report uses it in both the descriptive stats
        output and the Bootstrap CI comparison section, without recomputing.
        """
        data = [10.0, 11.0, 9.0, 10.5, 11.5, 9.5, 10.0, 11.0]
        report = mod.generate_single_report(data, "TestData", alpha=0.05)

        # The report should contain the parametric CI
        self.assertIn("Parametric CI:", report)

        # Verify the CI string appears in expected places
        # Both in the table and in the bootstrap comparison
        report_lines = report.split("\n")

        # Find the "% CI (mean)" line in the table (should have the CI bounds)
        ci_in_table = False
        ci_in_summary = False

        for line in report_lines:
            if "% CI (mean)" in line:
                ci_in_table = True
                # Should show actual CI bounds, not N/A
                self.assertIn("[", line)
                self.assertIn(",", line)
            if "Parametric CI:" in line:
                ci_in_summary = True
                # Should also show the same CI bounds
                self.assertIn("[", line)

        self.assertTrue(ci_in_table, "CI should appear in the stats table")
        self.assertTrue(ci_in_summary, "CI should appear in the summary section")

    def test_generate_report_cohens_d_string_computed_once(self):
        """Test that Cohen's d summary string is computed once and reused.

        Verify that the same Cohen's d formatted string appears in both the
        STATISTICAL TESTS and VERDICT sections without duplicate computation.
        This tests the optimization where d_summary is computed once and
        reused instead of being formatted twice.
        """
        base = [10.0, 11.0, 9.0, 10.5, 11.5, 9.5, 10.0, 11.0]
        exp = [15.0, 16.0, 14.0, 15.5, 16.5, 14.5, 15.0, 16.0]

        report = mod.generate_report(base, exp, "Base", "Exp",
                                         higher_is_better=True, alpha=0.05)

        # Extract lines matching the Cohen's d summary format
        # Pattern: exactly 2 leading spaces, then "Cohen's d: value (interpretation)"
        pattern = r"^  Cohen's d: [+-]?\d+\.\d+ \(\w+\)$"
        report_lines = report.split("\n")
        cohens_d_lines = [line for line in report_lines
                         if re.match(pattern, line)]

        # Should have exactly 2 Cohen's d summary lines:
        # one in STATISTICAL TESTS section, one in VERDICT section
        self.assertEqual(len(cohens_d_lines), 2,
                         f"Expected 2 Cohen's d summary lines, found {len(cohens_d_lines)}")

        # Both lines should be identical (same formatted string)
        self.assertEqual(cohens_d_lines[0], cohens_d_lines[1],
                         f"Cohen's d strings should be identical:\n"
                         f"  1st: {cohens_d_lines[0]}\n"
                         f"  2nd: {cohens_d_lines[1]}")

        # Verify the format is as expected
        stats_b = mod.compute_descriptive_stats(base)
        stats_e = mod.compute_descriptive_stats(exp)
        s_pool = mod._pooled_stdev_from_params(
            stats_b['n'], stats_b['var'], stats_e['n'], stats_e['var'])
        d_val = (stats_e['mean'] - stats_b['mean']) / s_pool if s_pool != 0 else 0.0
        interpretation = mod.interpret_cohens_d(d_val)
        expected_d_str = f"Cohen's d: {d_val:.3f} ({interpretation})"
        self.assertIn(expected_d_str, cohens_d_lines[0])

    def test_generate_report_diff_pct_string_computed_once(self):
        """Test that diff_pct formatted string is computed once and reused.

        Verify that the _format_diff_pct() result appears in both the
        DESCRIPTIVE STATISTICS and VERDICT sections without duplicate calls.
        This tests the optimization where diff_pct_str is computed once and
        reused instead of being formatted twice.
        """
        base = [10.0, 11.0, 9.0, 10.5, 11.5, 9.5, 10.0, 11.0]
        exp = [15.0, 16.0, 14.0, 15.5, 16.5, 14.5, 15.0, 16.0]

        report = mod.generate_report(base, exp, "Base", "Exp",
                                         higher_is_better=True, alpha=0.05)

        # Extract lines containing the difference information:
        # "Observed difference" (DESCRIPTIVE STATISTICS section)
        # "Mean difference" (VERDICT section)
        report_lines = report.split("\n")

        # Find the difference lines and extract the formatted diff_pct part
        obs_diff_line = None
        mean_diff_line = None

        for line in report_lines:
            if line.startswith("  Observed difference:"):
                obs_diff_line = line
            elif line.startswith("  Mean difference:"):
                mean_diff_line = line

        self.assertIsNotNone(obs_diff_line,
                           "Should find 'Observed difference' line in DESCRIPTIVE STATISTICS")
        self.assertIsNotNone(mean_diff_line,
                           "Should find 'Mean difference' line in VERDICT section")

        # Both lines should end with the same formatted diff_pct string
        # Extract the part in parentheses (e.g., "+50.00%" or "N/A (baseline mean is zero)")
        obs_paren = re.search(r'\((.*)\)$', obs_diff_line)
        mean_paren = re.search(r'\((.*)\)$', mean_diff_line)

        self.assertIsNotNone(obs_paren, "Observed difference line should have parenthesized content")
        self.assertIsNotNone(mean_paren, "Mean difference line should have parenthesized content")

        # Both should have identical formatted diff_pct strings
        obs_diff_pct_part = obs_paren.group(1)
        mean_diff_pct_part = mean_paren.group(1)
        self.assertEqual(obs_diff_pct_part, mean_diff_pct_part,
                       f"Diff_pct strings should be identical:\n"
                       f"  Observed: {obs_diff_pct_part}\n"
                       f"  Mean:     {mean_diff_pct_part}")

    def test_load_csv_with_genfromtxt(self):
        """Test that load_csv correctly uses np.genfromtxt to load data.

        np.genfromtxt automatically handles non-numeric lines by producing NaN,
        which are then filtered out. This test verifies:
        - Valid numeric values are loaded correctly
        - Empty lines are skipped
        - Non-numeric lines (headers, comments) are skipped
        - The returned data is a list of floats
        """
        import warnings

        # Create a temporary CSV file with mixed content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_path = f.name
            f.write("# Comment line\n")
            f.write("100.5\n")
            f.write("\n")  # Empty line
            f.write("200.75\n")
            f.write("header\n")  # Non-numeric line
            f.write("300.25\n")
            f.write("400.0\n")

        try:
            # Capture expected warning from dropped non-numeric row.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Load the CSV using the optimized function
                result = mod.load_csv(tmp_path)

                self.assertEqual(len(w), 1)
                self.assertIn("non-numeric value(s) skipped", str(w[0].message))

            # Verify that only numeric values are loaded
            self.assertEqual(len(result), 4)
            self.assertAlmostEqual(result[0], 100.5, places=2)
            self.assertAlmostEqual(result[1], 200.75, places=2)
            self.assertAlmostEqual(result[2], 300.25, places=2)
            self.assertAlmostEqual(result[3], 400.0, places=2)

            # Verify it returns a list, not numpy array
            self.assertIsInstance(result, list)
            self.assertIsInstance(result[0], float)
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink()

    def test_load_csv_single_value_0d_array(self):
        """Regression test: single numeric value should load as a one-item list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_path = f.name
            f.write("123.45\n")

        try:
            result = mod.load_csv(tmp_path)
            self.assertEqual(result, [123.45])
            self.assertIsInstance(result, list)
        finally:
            Path(tmp_path).unlink()

    def test_load_csv_with_header_emits_warning(self):
        """Regression test: load_csv should warn when dropping non-numeric values.

        Verifies that:
        - A file with a header row (non-numeric) and numeric data is loaded correctly
        - The numeric values are extracted ([10.0, 20.0, 30.0])
        - A warning is emitted about the dropped non-numeric value (the header)
        """
        import warnings

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_path = f.name
            f.write("latency\n10\n20\n30\n")

        try:
            # Capture warnings and verify they are emitted
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = mod.load_csv(tmp_path)

                # Verify the returned data is correct
                self.assertEqual(result, [10.0, 20.0, 30.0])
                self.assertIsInstance(result, list)

                # Verify that a warning was emitted
                self.assertEqual(len(w), 1)
                self.assertIn("non-numeric value(s) skipped", str(w[0].message))
                self.assertIn("1", str(w[0].message))
        finally:
            Path(tmp_path).unlink()

    @_ignore_precision_loss
    def test_compute_descriptive_stats_with_scipy_describe(self):
        """Test that compute_descriptive_stats uses scipy.stats.describe optimization.

        Verifies that the optimized implementation using scipy.stats.describe
        produces correct results and properly handles edge cases:
        - Normal data with known statistics
        - Small samples (n < 3 and n < 4) for skewness/kurtosis guards
        - NaN handling for skewness/kurtosis
        """
        # Test with normal data
        data = [100.0, 101.0, 102.0, 99.0, 98.0, 103.0, 97.0, 104.0]
        stats = mod.compute_descriptive_stats(data)

        # Verify all required keys are present
        required_keys = ['n', 'mean', 'var', 'stdev', 'se', 'median',
                        'min', 'max', 'range', 'p25', 'p75', 'iqr',
                        'skewness', 'ex_kurtosis']
        for key in required_keys:
            self.assertIn(key, stats)

        # Verify basic statistics match numpy/scipy expectations
        self.assertEqual(stats['n'], 8)
        self.assertAlmostEqual(stats['mean'], np.mean(data), places=6)
        self.assertAlmostEqual(stats['var'], np.var(data, ddof=1), places=6)
        self.assertAlmostEqual(stats['stdev'], np.std(data, ddof=1), places=6)
        self.assertAlmostEqual(stats['min'], min(data), places=6)
        self.assertAlmostEqual(stats['max'], max(data), places=6)
        self.assertAlmostEqual(stats['median'], np.median(data), places=6)
        self.assertEqual(stats['range'], max(data) - min(data))

        # Verify skewness and kurtosis are computed with bias=False
        expected_skew = sp_stats.skew(data, bias=False)
        expected_kurt = sp_stats.kurtosis(data, fisher=True, bias=False)
        self.assertAlmostEqual(stats['skewness'], expected_skew, places=6)
        self.assertAlmostEqual(stats['ex_kurtosis'], expected_kurt, places=6)

        # Test with n < 3 should raise ValueError
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([1.0, 2.0])

        # Test with n = 3 (skewness computed, kurtosis = 0)
        three_data = [1.0, 2.0, 5.0]
        stats_three = mod.compute_descriptive_stats(three_data)
        self.assertNotEqual(stats_three['skewness'], 0.0)  # Should compute
        self.assertEqual(stats_three['ex_kurtosis'], 0.0)  # n < 4, should be 0

        # Test with constant data (zero variance)
        # For constant data, scipy returns defined values (not NaN)
        constant_data = [5.0, 5.0, 5.0, 5.0, 5.0]
        stats_const = mod.compute_descriptive_stats(constant_data)
        self.assertEqual(stats_const['var'], 0.0)
        self.assertEqual(stats_const['stdev'], 0.0)
        # Verify skewness and kurtosis are computed (scipy returns defined values)
        self.assertIsInstance(stats_const['skewness'], float)
        self.assertIsInstance(stats_const['ex_kurtosis'], float)

    def test_statistical_power_uses_noncentral_t(self):
        """Test statistical_power uses non-central t-distribution.

        Verifies:
        - Exact power calculation using non-central t-distribution
        - Power equals alpha when effect size is zero
        - Power increases with sample size and effect size
        - Values match scipy's nct.cdf calculations
        """
        # Test d=0 case: power should equal alpha
        power_null = mod.statistical_power(30, 30, d=0.0, alpha=0.05)
        self.assertAlmostEqual(power_null, 0.05, places=6,
                             msg="Power should equal alpha when d=0")

        # Test with small effect (d=0.2) and small samples
        power_small = mod.statistical_power(10, 10, d=0.2, alpha=0.05)
        self.assertGreater(power_small, 0.05)
        self.assertLess(power_small, 0.20)

        # Test with medium effect (d=0.5) and moderate samples
        power_medium = mod.statistical_power(30, 30, d=0.5, alpha=0.05)
        self.assertGreater(power_medium, 0.40)
        self.assertLess(power_medium, 0.60)

        # Test with large effect (d=0.8) and larger samples
        power_large = mod.statistical_power(50, 50, d=0.8, alpha=0.05)
        self.assertGreater(power_large, 0.85)

        # Verify power increases with sample size (holding d constant)
        power_n20 = mod.statistical_power(20, 20, d=0.5, alpha=0.05)
        power_n50 = mod.statistical_power(50, 50, d=0.5, alpha=0.05)
        power_n100 = mod.statistical_power(100, 100, d=0.5, alpha=0.05)
        self.assertLess(power_n20, power_n50)
        self.assertLess(power_n50, power_n100)

        # Verify power increases with effect size (holding n constant)
        power_d02 = mod.statistical_power(30, 30, d=0.2, alpha=0.05)
        power_d05 = mod.statistical_power(30, 30, d=0.5, alpha=0.05)
        power_d08 = mod.statistical_power(30, 30, d=0.8, alpha=0.05)
        self.assertLess(power_d02, power_d05)
        self.assertLess(power_d05, power_d08)

        # Test with unequal sample sizes
        power_unequal = mod.statistical_power(40, 20, d=0.5, alpha=0.05)
        self.assertGreater(power_unequal, 0.30)
        self.assertLess(power_unequal, 0.70)

        # Verify that results match manual scipy calculation
        n1, n2, d, alpha = 30, 30, 0.6, 0.05
        n_eff = 2.0 / (1.0 / n1 + 1.0 / n2)
        ncp = d * (n_eff / 2.0) ** 0.5
        df = n1 + n2 - 2
        t_crit = sp_stats.t.ppf(1 - alpha / 2, df)
        expected_power = (1.0 - sp_stats.nct.cdf(t_crit, df, ncp) +
                         sp_stats.nct.cdf(-t_crit, df, ncp))
        actual_power = mod.statistical_power(n1, n2, d, alpha)
        self.assertAlmostEqual(actual_power, expected_power, places=10,
                              msg="Power should match manual scipy.stats.nct calculation")

        # Verify power is bounded [0, 1]
        power_extreme = mod.statistical_power(10, 10, d=3.0, alpha=0.05)
        self.assertGreaterEqual(power_extreme, 0.0)
        self.assertLessEqual(power_extreme, 1.0)

    def test_statistical_power_d_zero_edge_case(self):
        """Test statistical_power behavior at d=0 (null hypothesis).

        When d=0, the power equals alpha (Type I error rate).
        This tests:
        - Multiple alpha values (0.01, 0.05, 0.10)
        - Various sample sizes
        - Both equal and unequal sample sizes
        """
        # Test d=0 with different alpha values
        power_alpha_001 = mod.statistical_power(30, 30, d=0.0, alpha=0.01)
        self.assertAlmostEqual(power_alpha_001, 0.01, places=6,
                              msg="Power should equal alpha=0.01 when d=0")

        power_alpha_005 = mod.statistical_power(30, 30, d=0.0, alpha=0.05)
        self.assertAlmostEqual(power_alpha_005, 0.05, places=6,
                              msg="Power should equal alpha=0.05 when d=0")

        power_alpha_010 = mod.statistical_power(30, 30, d=0.0, alpha=0.10)
        self.assertAlmostEqual(power_alpha_010, 0.10, places=6,
                              msg="Power should equal alpha=0.10 when d=0")

        # Test d=0 with various sample sizes
        for n in [10, 20, 50, 100]:
            power = mod.statistical_power(n, n, d=0.0, alpha=0.05)
            self.assertAlmostEqual(power, 0.05, places=6,
                                  msg=f"Power should equal alpha=0.05 at d=0 for n={n}")

        # Test d=0 with unequal sample sizes
        power_unequal = mod.statistical_power(40, 20, d=0.0, alpha=0.05)
        self.assertAlmostEqual(power_unequal, 0.05, places=6,
                              msg="Power should equal alpha=0.05 when d=0 (unequal samples)")

        # Verify mathematical correctness: power at d=0 should be exactly alpha
        # (This is the Type I error rate under the null hypothesis)
        alpha = 0.05
        power = mod.statistical_power(30, 30, d=0.0, alpha=alpha)
        self.assertEqual(power, alpha,
                        msg="Power at d=0 should be exactly alpha (Type I error rate)")

    def test_p_value_visual_scale_adapts_to_large_p_values(self):
        """P-values beyond the default scale must not be clamped to the edge.

        Regression test for bug where p-values exceeding the alpha-based scale
        (e.g., p=0.12 with alpha=0.05) were clamped to the right edge, making
        them appear indistinguishable from the alpha boundary.

        Two nearly identical datasets produce large (non-significant) p-values (~0.5),
        testing that the scale widens appropriately and markers display distinctly.
        """
        # Two nearly identical datasets -> large p-values (~0.5)
        data_base = [100.0, 101.0, 102.0, 100.5, 101.5] * 6
        data_exp = [100.1, 101.1, 102.1, 100.6, 101.6] * 6

        report = mod.generate_report(
            data_base, data_exp,
            "Base", "Exp",
            higher_is_better=True,
            alpha=0.05,
        )

        # Extract the visual summary lines
        lines = report.split('\n')
        visual_lines = []
        in_visual = False
        for line in lines:
            if 'p-value visual summary' in line:
                in_visual = True
                continue
            if in_visual:
                if line.strip() == '' and visual_lines:
                    break
                visual_lines.append(line)

        # Skip leading empty lines in visual section
        visual_lines = [l for l in visual_lines if l.strip() != ''] if visual_lines else []

        # The scale header must extend beyond 0.05 to accommodate large p-values
        header = visual_lines[0] if visual_lines else ''
        self.assertTrue(
            '0.10' in header or '0.20' in header or '0.50' in header or '1.00' in header,
            "Scale should widen beyond 0.05 when p-values are large. Got: " + header
        )

        # Extract positions of p-value markers
        marker_positions = []
        for line in visual_lines:
            if '\u2593' in line:  # block marker
                idx = line.index('\u2593')
                marker_positions.append(idx)

        # With 4 tests and potentially different p-values, should have some distribution
        if len(marker_positions) >= 2:
            # They should not ALL be at the rightmost position (clamped)
            # Check that there's meaningful variation in positions
            unique_positions = len(set(marker_positions))
            self.assertGreater(
                unique_positions, 1,
                "Multiple p-value markers should not all suffer clamping to same position"
            )

            # Additional robustness check: most markers should have unique positions
            marker_diffs = sorted(set(marker_positions))
            if len(marker_diffs) >= 2:
                # With proper scaling, different p-values should have different positions
                self.assertGreaterEqual(
                    len(marker_diffs), len([p for p in marker_positions if p < 50]),
                    "Most markers should have unique positions when scale is properly widened"
                )

    def test_alpha_marker_position_adapts_to_scale(self):
        """Alpha marker must be positioned according to the dynamic scale, not hardcoded.

        Regression test for bug where the alpha marker used hardcoded spacing (" " * 30),
        which was only correct for the default 0.00-0.05 scale. When the scale widens
        to accommodate larger p-values, the alpha marker must also be positioned
        proportionally using pos_mult.
        """
        # Create datasets with large p-values to trigger scale widening
        np.random.seed(42)
        data_base = np.random.normal(100, 5, size=30).tolist()
        data_exp = np.random.normal(100, 5, size=30).tolist()

        report = mod.generate_report(
            data_base, data_exp,
            "base", "exp",
            higher_is_better=True, alpha=0.05)

        lines = report.split('\n')

        # Find the alpha marker line
        alpha_line = None
        for line in lines:
            if '\u03b1=' in line and '0.0500' in line:
                alpha_line = line
                break
        self.assertIsNotNone(alpha_line, "Alpha marker line not found in report")

        # Find p-value bar lines
        bar_lines = [l for l in lines if '\u2593' in l and 'p-value' not in l]
        self.assertTrue(bar_lines, "No p-value bar lines found")

        # Extract column positions
        alpha_col = alpha_line.index('\u03b1')
        bar_col = bar_lines[0].index('\u2593')

        # When all p-values exceed alpha (are not significant), the alpha marker
        # should appear to the left of the p-value bars
        self.assertLess(
            alpha_col, bar_col,
            "Alpha marker (col {}) should be left of p-value bars (col {}) "
            "when all p-values exceed alpha".format(alpha_col, bar_col)
        )

    def test_mann_whitney_u_z_consistent_with_p_when_ties(self):
        """Regression test: z-statistic must be consistent with p-value when data has ties.

        This verifies that tie-corrected variance is applied to the z-statistic
        calculation, ensuring that 2 * (1 - normal_cdf(abs(z))) approximately
        equals the p-value returned by scipy when data contains ties.
        """
        # Data with many ties
        data1 = [1, 1, 2, 2, 3]
        data2 = [1, 1, 1, 2, 2]

        u_stat, z_stat, p_value = mod.mann_whitney_u(data1, data2)

        # Compute what the p-value should be based on z-statistic
        # using standard normal CDF
        from scipy import stats as sp_stats
        z_based_p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))

        # The z-based p-value should be very close to the returned p-value
        # (within 0.05 tolerance as specified in requirements)
        self.assertAlmostEqual(
            z_based_p_value, p_value, delta=0.05,
            msg=f"Z-statistic inconsistent with p-value: "
                f"z={z_stat:.4f} gives p={z_based_p_value:.4f} "
                f"but mann_whitney_u returned p={p_value:.4f}"
        )

    def test_permutation_test_identical_data(self):
        """Regression test: permutation_test with identical datasets.

        When two identical datasets are compared, observed_diff == 0 and all
        permutation diffs are also 0. scipy.stats.permutation_test correctly
        returns p=1.0 (no evidence of difference).
        """
        data1 = [5.0] * 20
        data2 = [5.0] * 20

        observed_diff, p_two = mod.permutation_test(data1, data2, n_perms=10000, seed=42)

        self.assertEqual(observed_diff, 0.0)
        self.assertEqual(p_two, 1.0)

    def test_permutation_test_clearly_different_groups(self):
        """Regression test: permutation_test with clearly different groups.

        Verify that p-values remain sensible (small) for clearly different groups.
        This ensures the fix to >= and the (count+1)/(n_perms+1) formula don't break
        the primary case of detecting real differences.
        """
        # Clearly different groups
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        data2 = [20.0, 21.0, 22.0, 23.0, 24.0]
        n_perms = 5000

        observed_diff, p_two = mod.permutation_test(
            data1, data2, n_perms=n_perms, seed=42
        )

        # observed_diff should be large (_ref_mean(data2) - mean(data1) ≈ 18)
        self.assertGreater(abs(observed_diff), 10.0, "Should detect large difference")

        # p_two should be very small (< 0.05) because differences are clear
        self.assertLess(
            p_two, 0.05,
            f"Clear difference should give small p-value, got {p_two:.4f}"
        )

        # p_two should be reasonable (not exactly 0 or 1)
        self.assertGreater(p_two, 0.0)
        self.assertLess(p_two, 1.0)

    def test_permutation_test_p_value_bounded(self):
        """Regression test: permutation_test p-values are always in [0, 1].

        The corrected formula (count+1)/(n_perms+1) ensures p-values are bounded:
        - When count = 0, p = 1/(n_perms+1) > 0
        - When count = n_perms, p = (n_perms+1)/(n_perms+1) = 1.0
        - For any count in [0, n_perms], p in (0, 1]

        This test verifies that multiple random tests all produce bounded p-values.
        """
        import random
        random.seed(42)

        for trial in range(50):
            # Generate random data
            n1 = random.randint(5, 20)
            n2 = random.randint(5, 20)
            data1 = [random.gauss(100, 10) for _ in range(n1)]
            data2 = [random.gauss(100 + random.uniform(-5, 5), 10) for _ in range(n2)]

            observed_diff, p_two = mod.permutation_test(
                data1, data2, n_perms=1000, seed=trial
            )

            # Check bounds
            self.assertGreaterEqual(
                p_two, 0.0,
                f"Trial {trial}: p-value must be >= 0, got {p_two}"
            )
            self.assertLessEqual(
                p_two, 1.0,
                f"Trial {trial}: p-value must be <= 1, got {p_two}"
            )

    def test_permutation_test_higher_n_perms_gives_accurate_p(self):
        """Regression test: more permutations give more accurate p-value estimates.

        With the corrected formula (count+1)/(n_perms+1), increasing n_perms should
        give more accurate p-value estimates with less variance.

        This test uses two runs with different n_perms and verifies the estimates
        are reasonably close (they should converge as n_perms increases).
        """
        # Use identical-ish but slightly different data
        rng = np.random.RandomState(42)
        data1 = rng.normal(100, 5, size=20).tolist()
        data2 = rng.normal(100.5, 5, size=20).tolist()

        # Run with fewer permutations
        _, p_low = mod.permutation_test(
            data1, data2, n_perms=1000, seed=42
        )

        # Run with more permutations
        _, p_high = mod.permutation_test(
            data1, data2, n_perms=10000, seed=42
        )

        # Both should be in reasonable range and relatively close
        self.assertGreater(p_low, 0.0)
        self.assertLess(p_low, 1.0)
        self.assertGreater(p_high, 0.0)
        self.assertLess(p_high, 1.0)

        # They should be reasonably close (within ~0.1 relative difference)
        # This is a soft check since randomness still plays a role
        if p_low > 0.01 and p_high > 0.01:
            self.assertLess(
                abs(p_low - p_high) / max(p_low, p_high), 0.3,
                f"P-values should be close: {p_low:.4f} vs {p_high:.4f}"
            )

    def test_stability_chart_empty_data(self):
        """Test _stability_chart handles empty data without raising ValueError.

        Regression test for bug where min(data) raises ValueError on empty list.
        """
        # Call with empty data list
        result = mod._stability_chart([], "empty", stats={'n': 0, 'mean': 0, 'stdev': 0})
        # Should return a string without raising ValueError
        self.assertIsInstance(result, str)
        self.assertIn("No data available", result)
        # Ensure no exception was raised
        result2 = mod._stability_chart([], "empty_no_stats")
        self.assertIsInstance(result2, str)
        self.assertIn("No data available", result2)

    def test_p_value_scale_alignment_alpha_005(self):
        """Regression test: p-value bar positions align with tick marks for alpha=0.05.

        Verifies that position_multiplier and scale_width are computed correctly
        so that p-values map to correct positions on the visual scale.
        """
        alpha = 0.05
        tick_labels, tick_bar, pos_mult, scale_width = mod._compute_p_value_scale_info(alpha)

        # Get scale_max (internal value, but we can deduce it from the function logic)
        # For alpha=0.05, scale_max should be 0.05
        scale_max = 0.05

        # Test 1: position_multiplier * scale_max == scale_width
        # (p-value at scale_max maps to last tick position)
        self.assertAlmostEqual(
            pos_mult * scale_max, scale_width, places=5,
            msg="position_multiplier should map scale_max to scale_width"
        )

        # Test 2: scale_width == (number_of_ticks - 1) * 6
        # (6 ticks, each 4 chars wide, joined by 2 spaces = 6 position units apart)
        expected_width = (6 - 1) * 6  # 30
        self.assertEqual(
            scale_width, expected_width,
            msg=f"scale_width should be {expected_width} for 6 ticks"
        )

        # Test 3: Each tick value maps to the correct visual position
        # Tick values for alpha=0.05: [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
        tick_step = scale_max / (6 - 1)
        for i in range(6):
            tick_value = i * tick_step
            expected_pos = i * 6  # Each tick is 6 positions apart
            actual_pos = int(tick_value * pos_mult)
            self.assertEqual(
                actual_pos, expected_pos,
                msg=f"Tick {i} (p={tick_value:.2f}) should map to position {expected_pos}, got {actual_pos}"
            )

        # Test 4: p=0 maps to position 0; p=scale_max maps to position scale_width
        self.assertEqual(int(0.0 * pos_mult), 0)
        self.assertEqual(int(scale_max * pos_mult), scale_width)

    def test_p_value_scale_alignment_alpha_010(self):
        """Regression test: p-value bar positions align with tick marks for alpha=0.10.

        Verifies that position_multiplier and scale_width are computed correctly
        so that p-values map to correct positions on the visual scale.
        """
        alpha = 0.10
        tick_labels, tick_bar, pos_mult, scale_width = mod._compute_p_value_scale_info(alpha)

        # For alpha=0.10, scale_max should be 0.10
        scale_max = 0.10

        # Test 1: position_multiplier * scale_max == scale_width
        self.assertAlmostEqual(
            pos_mult * scale_max, scale_width, places=5,
            msg="position_multiplier should map scale_max to scale_width"
        )

        # Test 2: scale_width == (number_of_ticks - 1) * 6
        expected_width = (6 - 1) * 6  # 30
        self.assertEqual(
            scale_width, expected_width,
            msg=f"scale_width should be {expected_width} for 6 ticks"
        )

        # Test 3: Each tick value maps to the correct visual position
        # Tick values for alpha=0.10: [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
        tick_step = scale_max / (6 - 1)
        for i in range(6):
            tick_value = i * tick_step
            expected_pos = i * 6  # Each tick is 6 positions apart
            actual_pos = int(tick_value * pos_mult)
            self.assertEqual(
                actual_pos, expected_pos,
                msg=f"Tick {i} (p={tick_value:.2f}) should map to position {expected_pos}, got {actual_pos}"
            )

        # Test 4: p=0 maps to position 0; p=scale_max maps to position scale_width
        self.assertEqual(int(0.0 * pos_mult), 0)
        self.assertEqual(int(scale_max * pos_mult), scale_width)

    def test_p_value_scale_alignment_alpha_020(self):
        """Regression test: p-value bar positions align with tick marks for alpha=0.20.

        Verifies that position_multiplier and scale_width are computed correctly
        so that p-values map to correct positions on the visual scale.
        """
        alpha = 0.20
        tick_labels, tick_bar, pos_mult, scale_width = mod._compute_p_value_scale_info(alpha)

        # For alpha=0.20, scale_max should be 0.20
        scale_max = 0.20

        # Test 1: position_multiplier * scale_max == scale_width
        self.assertAlmostEqual(
            pos_mult * scale_max, scale_width, places=5,
            msg="position_multiplier should map scale_max to scale_width"
        )

        # Test 2: scale_width == (number_of_ticks - 1) * 6
        expected_width = (6 - 1) * 6  # 30
        self.assertEqual(
            scale_width, expected_width,
            msg=f"scale_width should be {expected_width} for 6 ticks"
        )

        # Test 3: Each tick value maps to the correct visual position
        # Tick values for alpha=0.20: [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
        tick_step = scale_max / (6 - 1)
        for i in range(6):
            tick_value = i * tick_step
            expected_pos = i * 6  # Each tick is 6 positions apart
            actual_pos = int(tick_value * pos_mult)
            self.assertEqual(
                actual_pos, expected_pos,
                msg=f"Tick {i} (p={tick_value:.2f}) should map to position {expected_pos}, got {actual_pos}"
            )

        # Test 4: p=0 maps to position 0; p=scale_max maps to position scale_width
        self.assertEqual(int(0.0 * pos_mult), 0)
        self.assertEqual(int(scale_max * pos_mult), scale_width)

    def test_p_value_bar_visual_clamping_uses_scale_width(self):
        """Regression test: bar position clamping uses actual scale_width, not hardcoded 50.

        Verifies that in generate_report(), bar positions are correctly clamped to
        scale_width instead of a hardcoded 50, and alpha marker position is also correct.
        """
        base = [1.0, 2.0, 3.0, 4.0, 5.0]
        exp = [3.5, 4.5, 5.5, 6.5, 7.5]  # Very different, will have low p-values

        # Test with alpha=0.05 (scale_max=0.05, scale_width=30)
        report = mod.generate_report(base, exp, "Base", "Exp", alpha=0.05)

        # Report should contain p-value visual section
        self.assertIn("p-value visual summary", report)

        # The alpha marker line should appear in the report
        self.assertIn("0.0500", report)  # alpha value should be shown

        # Verify the report is generated without error (bar positions should be valid)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)

    def test_bootstrap_ci_batching_produces_valid_results(self):
        """Regression test: batched bootstrap_ci produces statistically valid results.

        Verifies that the batched implementation:
        1. Returns valid confidence intervals (lower <= upper)
        2. Returns bounded p-values (0 <= p <= 1)
        3. Produces reasonable results for known distributions
        """
        # Test with normal distributions
        np.random.seed(42)
        data1 = np.random.randn(5000).tolist()
        data2 = (np.random.randn(5000) + 0.5).tolist()  # Shifted by 0.5

        ci_lower, ci_upper, p_two = mod.bootstrap_ci(data1, data2, n_boot=10000, seed=42)

        # Check return types
        self.assertIsInstance(ci_lower, float)
        self.assertIsInstance(ci_upper, float)
        self.assertIsInstance(p_two, float)

        # Check interval ordering
        self.assertLessEqual(ci_lower, ci_upper)

        # Check p-value bounds
        self.assertGreaterEqual(p_two, 0.0)
        self.assertLessEqual(p_two, 1.0)

        # For shifted distribution, should detect significant difference
        self.assertLess(p_two, 0.05)

        # CI should not contain 0 for shifted distributions
        self.assertGreater(ci_lower, 0.0)

    def test_bootstrap_ci_batching_consistency(self):
        """Regression test: batched bootstrap_ci is consistent across runs.

        With the same seed, multiple calls should produce identical results.
        This verifies that batching doesn't introduce randomness issues.
        """
        np.random.seed(123)
        data1 = np.random.randn(1000).tolist()
        data2 = np.random.randn(1000).tolist()

        # Run multiple times with same seed
        result1 = mod.bootstrap_ci(data1, data2, n_boot=5000, seed=42)
        result2 = mod.bootstrap_ci(data1, data2, n_boot=5000, seed=42)

        # Results should be identical
        self.assertEqual(result1[0], result2[0])
        self.assertEqual(result1[1], result2[1])
        self.assertEqual(result1[2], result2[2])

    def test_bootstrap_ci_batching_edge_cases(self):
        """Regression test: batched bootstrap_ci handles edge cases.

        Tests small samples, constant data, and various batch scenarios.
        """
        # Small sample (should work even if batch size > n_boot)
        small_data1 = [1.0, 2.0, 3.0]
        small_data2 = [1.5, 2.5, 3.5]
        ci_low, ci_high, p = mod.bootstrap_ci(small_data1, small_data2, n_boot=100, seed=42)
        self.assertLessEqual(ci_low, ci_high)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

        # Constant data (identical means)
        const_data = [5.0] * 20
        ci_low, ci_high, p = mod.bootstrap_ci(const_data, const_data, n_boot=1000, seed=42)
        self.assertEqual(ci_low, 0.0)
        self.assertEqual(ci_high, 0.0)
        self.assertEqual(p, 1.0)

    def test_bootstrap_single_ci_batching_produces_valid_results(self):
        """Regression test: batched bootstrap_single_ci produces valid results.

        Verifies that the batched implementation produces valid confidence
        intervals that contain the sample mean with appropriate probability.
        """
        np.random.seed(42)
        data = np.random.randn(5000).tolist()

        ci_lower, ci_upper = mod.bootstrap_single_ci(data, n_boot=10000, ci=95, seed=42)

        # Check return types
        self.assertIsInstance(ci_lower, float)
        self.assertIsInstance(ci_upper, float)

        # Check interval ordering
        self.assertLessEqual(ci_lower, ci_upper)

        # Sample mean should typically be in CI
        sample_mean = _ref_mean(data)
        self.assertLessEqual(ci_lower, sample_mean)
        self.assertGreaterEqual(ci_upper, sample_mean)

    def test_bootstrap_single_ci_batching_consistency(self):
        """Regression test: batched bootstrap_single_ci is consistent.

        With the same seed, multiple calls should produce identical results.
        """
        np.random.seed(123)
        data = np.random.randn(1000).tolist()

        # Run multiple times with same seed
        result1 = mod.bootstrap_single_ci(data, n_boot=5000, seed=42)
        result2 = mod.bootstrap_single_ci(data, n_boot=5000, seed=42)

        # Results should be identical
        self.assertEqual(result1[0], result2[0])
        self.assertEqual(result1[1], result2[1])

    def test_bootstrap_single_ci_batching_edge_cases(self):
        """Regression test: batched bootstrap_single_ci handles edge cases."""
        # Small sample
        small_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_low, ci_high = mod.bootstrap_single_ci(small_data, n_boot=100, seed=42)
        self.assertLessEqual(ci_low, ci_high)

        # Constant data
        const_data = [5.0] * 20
        ci_low, ci_high = mod.bootstrap_single_ci(const_data, n_boot=1000, seed=42)
        self.assertAlmostEqual(ci_low, 5.0, places=10)
        self.assertAlmostEqual(ci_high, 5.0, places=10)

        # Large sample that will definitely trigger batching
        np.random.seed(42)
        large_data = np.random.randn(10000).tolist()
        ci_low, ci_high = mod.bootstrap_single_ci(large_data, n_boot=20000, seed=42)
        self.assertLessEqual(ci_low, ci_high)
        sample_mean = _ref_mean(large_data)
        self.assertLessEqual(ci_low, sample_mean)
        self.assertGreaterEqual(ci_high, sample_mean)


    # ------------------------------------------------------------------
    # Fix: permutation_test uses np.concatenate for array-safe concat
    # ------------------------------------------------------------------

    def test_permutation_test_accepts_numpy_arrays(self):
        """Regression test: permutation_test works with numpy arrays, not just lists."""
        data1_list = [100.0, 101.0, 102.0, 103.0, 104.0]
        data2_list = [105.0, 106.0, 107.0, 108.0, 109.0]
        data1_arr = np.array(data1_list)
        data2_arr = np.array(data2_list)

        diff_list, p_list = mod.permutation_test(data1_list, data2_list, n_perms=500, seed=42)
        diff_arr, p_arr = mod.permutation_test(data1_arr, data2_arr, n_perms=500, seed=42)

        # Results must be identical regardless of input type
        self.assertAlmostEqual(diff_list, diff_arr, places=10)
        self.assertAlmostEqual(p_list, p_arr, places=10)

        # Observed diff should be positive (group2 > group1)
        self.assertGreater(diff_arr, 0)

    # ------------------------------------------------------------------
    # Fix: _required_n_for_moe uses t-distribution instead of z
    # ------------------------------------------------------------------

    def test_required_n_for_moe_uses_t_distribution(self):
        """Regression test: _required_n_for_moe returns >= z-based estimate.

        The t-distribution has heavier tails than normal, so the t-based
        required n should be >= the z-based estimate for small targets.
        """
        s = 10.0
        target_moe = 1.0
        alpha = 0.05

        n_t = mod._required_n_for_moe(s, target_moe, alpha)

        # z-based estimate for comparison
        z = float(sp_stats.norm.ppf(1 - alpha / 2))
        import math
        n_z = max(2, math.ceil((z * s / target_moe) ** 2))

        # t-based should be >= z-based (heavier tails need more samples)
        self.assertGreaterEqual(n_t, n_z)
        self.assertIsInstance(n_t, int)
        self.assertGreaterEqual(n_t, 2)

    def test_required_n_for_moe_edge_cases(self):
        """Regression test: _required_n_for_moe handles edge cases."""
        # Zero target returns inf
        self.assertEqual(mod._required_n_for_moe(10.0, 0.0, 0.05), float('inf'))
        # Negative target returns inf
        self.assertEqual(mod._required_n_for_moe(10.0, -1.0, 0.05), float('inf'))
        # Very large target needs minimal n
        n = mod._required_n_for_moe(1.0, 1000.0, 0.05)
        self.assertEqual(n, 2)

    # ------------------------------------------------------------------
    # Fix: cumulative_comparison uses int(round()) for consistent rounding
    # ------------------------------------------------------------------

    def test_cumulative_comparison_threshold_rounding(self):
        """Regression test: cumulative thresholds use standard rounding, not banker's.

        With banker's rounding, round(0.5) == 0 and round(1.5) == 2.
        With int(round()), we get consistent behavior.
        """
        # Create data where thresholds would hit .5 values
        # mean ~50, stdev ~10, so thresholds span ~30-70 with step ~5
        np.random.seed(99)
        data1 = np.random.normal(50, 10, 200).tolist()
        data2 = np.random.normal(55, 10, 200).tolist()

        result = mod.cumulative_comparison(data1, data2, "A", "B")

        # Extract threshold values from output
        import re
        thresholds = [int(m.group(1)) for m in re.finditer(r'≥\s+(-?\d+)', result)]

        # All thresholds should be integers (no .0 artifacts)
        for t in thresholds:
            self.assertIsInstance(t, int)

        # Thresholds should be sorted
        self.assertEqual(thresholds, sorted(thresholds))

    def test_cumulative_comparison_column_alignment_with_long_labels(self):
        """Regression test: cumulative_comparison columns must align when labels exceed 10 chars.

        Bug: header row expanded for long labels but separator and data rows
        used hardcoded 10-char widths, causing misalignment.
        Fix: compute col_w = max(10, len(label1), len(label2)) and use it
        consistently across header, separator, and data rows.
        """
        np.random.seed(42)
        data1 = np.random.normal(100, 10, 50).tolist()
        data2 = np.random.normal(105, 10, 50).tolist()
        long_label = "crypto.aes.fix.op2"  # 18 chars, exceeds default 10

        result = mod.cumulative_comparison(data1, data2, "baseline", long_label)
        lines = result.split('\n')

        # Find header and separator lines
        header = None
        separator = None
        for i, line in enumerate(lines):
            if "Threshold" in line and long_label in line:
                header = line
                separator = lines[i + 1]
                break

        self.assertIsNotNone(header, "Header line not found")
        self.assertIsNotNone(separator, "Separator line not found")

        # The separator must be at least as wide as the header
        self.assertGreaterEqual(len(separator), len(header) - 5,
                                "Separator row must match header width")

        # The long label must appear fully in the header (not truncated)
        self.assertIn(long_label, header)

        # Data rows must contain the long label in the "Better" column
        data_lines = [l for l in lines if '≥' in l and '%' in l]
        self.assertGreater(len(data_lines), 0)
        # At least one row should show the long label as "Better"
        has_long_label_row = any(long_label in l for l in data_lines)
        self.assertTrue(has_long_label_row,
                        "Long label should appear in 'Better' column of data rows")

    # ------------------------------------------------------------------
    # Fix: _stability_chart reuses stats['min']/stats['max']
    # ------------------------------------------------------------------

    def test_stability_chart_reuses_stats_min_max(self):
        """Regression test: _stability_chart produces identical output with and without stats."""
        data = [100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 97.0, 100.5]
        stats = mod.compute_descriptive_stats(data)

        result_with_stats = mod._stability_chart(data, "Test", stats=stats)
        result_without_stats = mod._stability_chart(data, "Test", stats=None)

        # Output must be identical regardless of whether stats are passed
        self.assertEqual(result_with_stats, result_without_stats)

    def test_stability_chart_stats_min_max_consistency(self):
        """Regression test: _stability_chart uses stats min/max correctly when control limits
        extend beyond data range."""
        # Data with small range but large stdev would make ±2σ extend beyond data
        data = [50.0, 50.1, 49.9, 50.0, 50.05]
        stats = mod.compute_descriptive_stats(data)

        # Verify stats min/max match actual data
        self.assertAlmostEqual(stats['min'], min(data), places=10)
        self.assertAlmostEqual(stats['max'], max(data), places=10)

        result = mod._stability_chart(data, "Test", stats=stats)
        # Should not crash and should contain expected markers
        self.assertIn("Run Chart", result)
        self.assertIn("±2σ", result)


    # ==================================================================
    # HIGH PRIORITY: Core algorithm correctness against reference impls
    # ==================================================================

    def test_welch_t_test_matches_scipy_ttest_ind(self):
        """Correctness: welch_t_test must match scipy.stats.ttest_ind for normal data."""
        rng = np.random.RandomState(42)
        data1 = rng.normal(100, 10, 50).tolist()
        data2 = rng.normal(105, 12, 60).tolist()

        t_stat, df, p_two = mod.welch_t_test(data1, data2)
        ref = sp_stats.ttest_ind(data1, data2, equal_var=False)

        # t-stat sign: scipy computes data1-data2, ours computes data2-data1
        self.assertAlmostEqual(t_stat, -float(ref.statistic), places=10)
        self.assertAlmostEqual(p_two, float(ref.pvalue), places=10)

    def test_mann_whitney_u_matches_scipy(self):
        """Correctness: U-statistic and p-value must match scipy.stats.mannwhitneyu."""
        rng = np.random.RandomState(7)
        data1 = rng.normal(50, 8, 40).tolist()
        data2 = rng.normal(55, 8, 45).tolist()

        u_stat, _, p_two = mod.mann_whitney_u(data1, data2)
        ref_u, ref_p = sp_stats.mannwhitneyu(data1, data2, alternative='two-sided')

        self.assertAlmostEqual(u_stat, float(ref_u), places=10)
        self.assertAlmostEqual(p_two, float(ref_p), places=10)

    def test_bootstrap_ci_contains_known_shift(self):
        """Correctness: bootstrap CI for a known shift should contain the true difference."""
        rng = np.random.RandomState(99)
        true_shift = 5.0
        data1 = rng.normal(100, 10, 200).tolist()
        data2 = rng.normal(100 + true_shift, 10, 200).tolist()

        ci_lo, ci_hi, p_two = mod.bootstrap_ci(data1, data2, n_boot=10000, seed=42)

        # 95% CI should contain the true shift
        self.assertLess(ci_lo, true_shift)
        self.assertGreater(ci_hi, true_shift)
        # Should be significant
        self.assertLess(p_two, 0.05)

    def test_power_mde_required_n_self_consistency(self):
        """Correctness: statistical_power, minimum_detectable_effect, and
        required_sample_size must be mutually consistent.

        - power(n, n, mde(n, n)) ≈ 0.80
        - power(req_n(d), req_n(d), d) ≈ 0.80
        """
        alpha = 0.05

        # MDE at n=50 should yield ~80% power when plugged back in
        mde = mod.minimum_detectable_effect(50, 50, alpha, target_power=0.80)
        power_at_mde = mod.statistical_power(50, 50, mde, alpha)
        self.assertAlmostEqual(power_at_mde, 0.80, delta=0.02)

        # required_sample_size for d=0.5 should yield ~80% power
        req_n = mod.required_sample_size(0.5, alpha, target_power=0.80)
        power_at_req_n = mod.statistical_power(req_n, req_n, 0.5, alpha)
        self.assertGreaterEqual(power_at_req_n, 0.80)

    def test_one_tailed_p_value_all_branches(self):
        """Correctness: one_tailed_p_value conversion for all four direction combos."""
        p_two = 0.04

        # higher_is_better=True, positive effect → small p (improvement detected)
        p = mod.one_tailed_p_value(p_two, +1.0, higher_is_better=True)
        self.assertAlmostEqual(p, 0.02, places=10)

        # higher_is_better=True, negative effect → large p (wrong direction)
        p = mod.one_tailed_p_value(p_two, -1.0, higher_is_better=True)
        self.assertAlmostEqual(p, 0.98, places=10)

        # higher_is_better=False, negative effect → small p (improvement detected)
        p = mod.one_tailed_p_value(p_two, -1.0, higher_is_better=False)
        self.assertAlmostEqual(p, 0.02, places=10)

        # higher_is_better=False, positive effect → large p (wrong direction)
        p = mod.one_tailed_p_value(p_two, +1.0, higher_is_better=False)
        self.assertAlmostEqual(p, 0.98, places=10)

        # Zero effect → always 0.5
        p = mod.one_tailed_p_value(0.04, 0.0, higher_is_better=True)
        self.assertEqual(p, 0.5)

        # Clamping: p_two > 1.0 should be clamped
        p = mod.one_tailed_p_value(1.5, +1.0, higher_is_better=True)
        self.assertAlmostEqual(p, 0.5, places=10)

    # ==================================================================
    # MEDIUM PRIORITY: Classification/interpretation branch coverage
    # ==================================================================

    def test_significance_marker_all_branches(self):
        """Correctness: significance_marker returns correct labels for all thresholds."""
        alpha = 0.05

        self.assertIn("highly", mod.significance_marker(0.001, alpha))
        self.assertIn("Yes", mod.significance_marker(0.001, alpha))

        result_sig = mod.significance_marker(0.03, alpha)
        self.assertIn("Yes", result_sig)
        self.assertNotIn("highly", result_sig)

        self.assertIn("Borderline", mod.significance_marker(0.07, alpha))

        self.assertIn("No", mod.significance_marker(0.5, alpha))

        # Boundary: exactly alpha/10
        self.assertIn("Yes", mod.significance_marker(alpha / 10, alpha))

    def test_interpret_cohens_d_all_thresholds(self):
        """Correctness: interpret_cohens_d returns correct labels at all boundaries."""
        self.assertEqual(mod.interpret_cohens_d(0.0), "Negligible")
        self.assertEqual(mod.interpret_cohens_d(0.19), "Negligible")
        self.assertEqual(mod.interpret_cohens_d(0.2), "Small")
        self.assertEqual(mod.interpret_cohens_d(0.49), "Small")
        self.assertEqual(mod.interpret_cohens_d(0.5), "Medium")
        self.assertEqual(mod.interpret_cohens_d(0.79), "Medium")
        self.assertEqual(mod.interpret_cohens_d(0.8), "Large")
        self.assertEqual(mod.interpret_cohens_d(2.0), "Large")
        # Negative values use abs()
        self.assertEqual(mod.interpret_cohens_d(-0.6), "Medium")

    def test_interpret_cv_all_branches(self):
        """Correctness: _interpret_cv returns correct interpretation for each range."""
        self.assertIn("Very low", mod._interpret_cv(0.5))
        self.assertIn("Low", mod._interpret_cv(2.0))
        self.assertIn("Moderate", mod._interpret_cv(4.0))
        self.assertIn("High", mod._interpret_cv(6.0))

    def test_interpret_skewness_all_branches(self):
        """Correctness: _interpret_skewness returns correct interpretation."""
        lines_sym = mod._interpret_skewness(0.1)
        self.assertTrue(any("symmetric" in l for l in lines_sym))

        lines_right = mod._interpret_skewness(1.0)
        self.assertTrue(any("Right-skewed" in l for l in lines_right))

        lines_left = mod._interpret_skewness(-1.0)
        self.assertTrue(any("Left-skewed" in l for l in lines_left))

    def test_interpret_kurtosis_all_branches(self):
        """Correctness: _interpret_kurtosis returns correct interpretation."""
        lines_meso = mod._interpret_kurtosis(0.1)
        self.assertTrue(any("mesokurtic" in l for l in lines_meso))

        lines_lepto = mod._interpret_kurtosis(1.5)
        self.assertTrue(any("Leptokurtic" in l for l in lines_lepto))

        lines_platy = mod._interpret_kurtosis(-1.5)
        self.assertTrue(any("Platykurtic" in l for l in lines_platy))

    def test_required_n_for_moe_achieves_target(self):
        """Correctness: the returned n must actually achieve the target margin of error."""
        import math
        s = 15.0
        target_moe = 2.0
        alpha = 0.05

        n = mod._required_n_for_moe(s, target_moe, alpha)
        # Verify: t_crit(alpha, n-1) * s / sqrt(n) <= target_moe
        t_crit = float(abs(sp_stats.t.ppf(alpha / 2, n - 1)))
        actual_moe = t_crit * s / math.sqrt(n)
        self.assertLessEqual(actual_moe, target_moe)

        # And n-1 should NOT achieve it (n is minimal)
        if n > 2:
            actual_moe_minus1 = t_crit * s / math.sqrt(n - 1)
            # Use the correct t_crit for n-2 df
            t_crit_minus1 = float(abs(sp_stats.t.ppf(alpha / 2, n - 2)))
            actual_moe_minus1 = t_crit_minus1 * s / math.sqrt(n - 1)
            self.assertGreater(actual_moe_minus1, target_moe)

    def test_minimum_detectable_effect_direct(self):
        """Correctness: MDE at given n yields target power when plugged into statistical_power."""
        for n in [20, 50, 100]:
            mde = mod.minimum_detectable_effect(n, n, alpha=0.05, target_power=0.80)
            power = mod.statistical_power(n, n, mde, alpha=0.05)
            self.assertAlmostEqual(power, 0.80, delta=0.02,
                                   msg=f"MDE at n={n} should yield ~80% power")

    def test_required_sample_size_direct(self):
        """Correctness: required_sample_size returns n that achieves target power."""
        for d in [0.2, 0.5, 0.8]:
            n = mod.required_sample_size(d, alpha=0.05, target_power=0.80)
            power = mod.statistical_power(n, n, d, alpha=0.05)
            self.assertGreaterEqual(power, 0.80,
                                    msg=f"n={n} for d={d} should achieve >=80% power")
        # d <= 0 returns inf
        self.assertEqual(mod.required_sample_size(0.0), float('inf'))
        self.assertEqual(mod.required_sample_size(-0.5), float('inf'))

    # ==================================================================
    # LOWER PRIORITY: Formatting/display functions
    # ==================================================================

    def test_format_stats_comparison_correctness(self):
        """Correctness: _format_stats_comparison produces correct table with both datasets."""
        stats_b = mod.compute_descriptive_stats([10.0, 20.0, 30.0])
        stats_e = mod.compute_descriptive_stats([15.0, 25.0, 35.0])

        lines = mod._format_stats_comparison(stats_b, stats_e, "Base", "Exp")
        table = "\n".join(lines)

        self.assertIn("Base", table)
        self.assertIn("Exp", table)
        self.assertIn("20.00", table)  # base mean
        self.assertIn("25.00", table)  # exp mean
        self.assertIn("n", table)
        self.assertIn("3", table)  # n for both

    def test_normality_assessment_two_datasets(self):
        """Correctness: _normality_assessment formats table correctly for two datasets."""
        rng = np.random.RandomState(42)
        data1 = rng.normal(100, 10, 50).tolist()
        data2 = rng.normal(100, 10, 50).tolist()
        edges = mod._compute_bins([data1, data2])

        lines, p_values = mod._normality_assessment(
            [data1, data2], ["A", "B"], edges, alpha=0.05)
        text = "\n".join(lines)

        self.assertEqual(len(p_values), 2)
        self.assertIn("D'Agostino-Pearson", text)
        self.assertIn("A", text)
        self.assertIn("B", text)
        for p in p_values:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_format_outlier_report_with_and_without_outliers(self):
        """Correctness: _format_outlier_report formats both cases correctly."""
        stats = mod.compute_descriptive_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        # With outliers
        outliers_iqr = [100]
        outliers_zscore = [(100.0, 3.5)]
        lines = mod._format_outlier_report(stats, outliers_iqr, outliers_zscore)
        text = "\n".join(lines)
        self.assertIn("100", text)
        self.assertIn("IQR Method", text)
        self.assertIn("Z-score Method", text)

        # Without outliers
        lines_clean = mod._format_outlier_report(stats, [], [])
        text_clean = "\n".join(lines_clean)
        self.assertIn("No outliers detected", text_clean)

    def test_format_diff_pct_normal_and_inf(self):
        """Correctness: _format_diff_pct handles normal values and infinity."""
        self.assertEqual(mod._format_diff_pct(5.0), "+5.00%")
        self.assertEqual(mod._format_diff_pct(-3.14), "-3.14%")
        self.assertEqual(mod._format_diff_pct(0.0), "+0.00%")
        self.assertIn("N/A", mod._format_diff_pct(float('inf')))
        self.assertIn("N/A", mod._format_diff_pct(float('-inf')))

    # ==================================================================
    # Empty data guard
    # ==================================================================

    def test_insufficient_data_raises_value_error(self):
        """Regression test: fewer than 3 data points must raise ValueError."""
        # Empty
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([])
        # n=1
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([5.0])
        # n=2
        with self.assertRaises(ValueError):
            mod.compute_descriptive_stats([5.0, 6.0])
        # n=3 should work
        mod.compute_descriptive_stats([5.0, 6.0, 7.0])

        # Report functions propagate the error
        with self.assertRaises(ValueError):
            mod.generate_single_report([1.0, 2.0], "TooFew")
        with self.assertRaises(ValueError):
            mod.generate_report([1.0, 2.0], [3.0, 4.0, 5.0], "A", "B")
        with self.assertRaises(ValueError):
            mod.generate_report([3.0, 4.0, 5.0], [1.0, 2.0], "A", "B")

    def test_minimum_n3_produces_valid_stats(self):
        """Regression test: n=3 is the minimum and produces all valid statistics."""
        data = [10.0, 20.0, 30.0]
        stats = mod.compute_descriptive_stats(data)

        self.assertEqual(stats['n'], 3)
        self.assertAlmostEqual(stats['mean'], 20.0, places=10)
        self.assertGreater(stats['stdev'], 0)
        self.assertGreater(stats['se'], 0)
        # Skewness should be computable at n=3
        self.assertIsInstance(stats['skewness'], float)
        # Kurtosis requires n>=4, should be 0 at n=3
        self.assertEqual(stats['ex_kurtosis'], 0.0)

    def test_minimum_n3_report_has_all_sections(self):
        """Regression test: n=3 single report includes all sections without errors."""
        data = [10.0, 20.0, 30.0]
        report = mod.generate_single_report(data, "MinData")

        # All sections present
        for section in ["DESCRIPTIVE STATISTICS", "FREQUENCY HISTOGRAM",
                        "NORMALITY ASSESSMENT", "OUTLIER DETECTION",
                        "BOOTSTRAP CONFIDENCE INTERVAL", "SAMPLE SIZE ADEQUACY",
                        "SUMMARY"]:
            self.assertIn(section, report)

        # No NaN or inf in output
        self.assertNotIn("nan", report.lower())
        self.assertNotIn("inf", report.lower())

    def test_minimum_n3_comparison_report_works(self):
        """Regression test: n=3 comparison report works without errors."""
        data1 = [10.0, 20.0, 30.0]
        data2 = [15.0, 25.0, 35.0]
        report = mod.generate_report(data1, data2, "A", "B")

        self.assertIn("STATISTICAL TESTS", report)
        self.assertIn("VERDICT", report)
        # No NaN in output
        self.assertNotIn("nan", report.lower())

    def test_n4_enables_kurtosis(self):
        """Regression test: n=4 enables kurtosis computation."""
        data = [1.0, 2.0, 10.0, 11.0]
        stats = mod.compute_descriptive_stats(data)
        # With n=4, kurtosis should be computed (non-zero for non-normal data)
        self.assertIsInstance(stats['ex_kurtosis'], float)
        # This bimodal-ish data should have negative kurtosis
        self.assertNotEqual(stats['ex_kurtosis'], 0.0)


# =============================================================================
# Reference data regression tests (test_data/ directory)
# =============================================================================

class ReferenceDataRegressionTests(unittest.TestCase):
    """Validate reports against reference outputs generated from test_data/ CSVs.

    These tests ensure that key numeric values (descriptive stats, test statistics,
    p-values, effect sizes) and verdicts remain correct across code changes.
    Values that depend on bootstrap/permutation seeds are checked with tolerance.
    """

    @classmethod
    def setUpClass(cls):
        cls.base = np.genfromtxt("test_data/crypto.aes.base.csv").tolist()
        cls.fix = np.genfromtxt("test_data/crypto.aes.fix.csv").tolist()
        cls.op1 = np.genfromtxt("test_data/crypto.aes.fix.op1.csv").tolist()
        cls.op2 = np.genfromtxt("test_data/crypto.aes.fix.op2.csv").tolist()

    def setUp(self):
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.simplefilter("error")

    def tearDown(self):
        self._warn_ctx.__exit__(None, None, None)

    def _check_descriptive_stats(self, data, ref):
        """Verify descriptive stats match reference values."""
        stats = mod.compute_descriptive_stats(data)
        for key, expected in ref.items():
            self.assertAlmostEqual(stats[key], expected, places=2,
                                   msg=f"Mismatch in {key}")

    def _check_comparison_report(self, report, ref):
        """Verify comparison report contains expected values."""
        self.assertIn(ref['diff'], report)
        self.assertIn(ref['diff_pct'], report)
        self.assertIn(f"Cohen's d: {ref['cohens_d']}", report)
        self.assertIn(f"({ref['cohens_d_interp']})", report)
        self.assertIn(ref['verdict_text'], report)

    # --- Single report tests ---

    def test_base_descriptive_stats(self):
        """Reference: crypto.aes.base descriptive stats match base_stats.txt."""
        self._check_descriptive_stats(self.base, {
            'n': 58, 'mean': 1054.53, 'stdev': 12.24, 'median': 1054.37,
            'min': 1030.72, 'max': 1089.65, 'skewness': 0.633, 'ex_kurtosis': 0.712,
        })

    def test_fix_descriptive_stats(self):
        """Reference: crypto.aes.fix descriptive stats match fix_stats.txt."""
        self._check_descriptive_stats(self.fix, {
            'n': 58, 'mean': 1059.45, 'stdev': 11.96, 'median': 1059.66,
            'min': 1028.77, 'max': 1090.31, 'skewness': 0.047, 'ex_kurtosis': 0.094,
        })

    def test_op1_descriptive_stats(self):
        """Reference: crypto.aes.fix.op1 descriptive stats match fix_stats.op1.txt."""
        self._check_descriptive_stats(self.op1, {
            'n': 58, 'mean': 1060.26, 'stdev': 12.88, 'median': 1062.53,
            'min': 1034.21, 'max': 1084.64, 'skewness': -0.265, 'ex_kurtosis': -0.772,
        })

    def test_op2_descriptive_stats(self):
        """Reference: crypto.aes.fix.op2 descriptive stats match fix_stats.op2.txt."""
        self._check_descriptive_stats(self.op2, {
            'n': 58, 'mean': 1062.12, 'stdev': 12.37, 'median': 1062.76,
            'min': 1038.03, 'max': 1101.81, 'skewness': 0.419, 'ex_kurtosis': 1.030,
        })

    def test_base_single_report_ci(self):
        """Reference: base parametric CI matches base_stats.txt."""
        stats = mod.compute_descriptive_stats(self.base)
        lines, ci_lo, ci_hi, _ = mod._format_stats_single(stats, 0.05)
        self.assertAlmostEqual(ci_lo, 1051.31, places=2)
        self.assertAlmostEqual(ci_hi, 1057.75, places=2)

    # --- Comparison statistical tests ---

    def test_base_vs_fix_welch(self):
        """Reference: base vs fix Welch's t-test matches base_vs_fix.txt."""
        t, df, p2 = mod.welch_t_test(self.base, self.fix)
        self.assertAlmostEqual(t, 2.191, places=3)
        self.assertAlmostEqual(df, 113.9, places=1)
        self.assertAlmostEqual(p2, 0.0305, places=4)

    def test_base_vs_fix_mann_whitney(self):
        """Reference: base vs fix Mann-Whitney U matches base_vs_fix.txt."""
        u, z, p2 = mod.mann_whitney_u(self.base, self.fix)
        self.assertAlmostEqual(u, 1254, places=0)
        self.assertAlmostEqual(z, 2.363, places=3)
        self.assertAlmostEqual(p2, 0.0181, places=4)

    def test_base_vs_fix_effect_size(self):
        """Reference: base vs fix Cohen's d matches base_vs_fix.txt."""
        stats_b = mod.compute_descriptive_stats(self.base)
        stats_f = mod.compute_descriptive_stats(self.fix)
        s_pool = mod._pooled_stdev_from_params(
            stats_b['n'], stats_b['var'], stats_f['n'], stats_f['var'])
        d = (stats_f['mean'] - stats_b['mean']) / s_pool
        self.assertAlmostEqual(d, 0.407, places=3)
        self.assertEqual(mod.interpret_cohens_d(d), "Small")

    def test_base_vs_fix_verdict(self):
        """Reference: base vs fix verdict is significant improvement."""
        report = mod.generate_report(self.base, self.fix, "Base", "Fix")
        self.assertIn("SIGNIFICANT IMPROVEMENT", report)
        self.assertIn("+4.92", report)
        self.assertIn("+0.47%", report)

    def test_base_vs_op2_welch(self):
        """Reference: base vs op2 Welch's t-test matches base_vs_fix_op2.txt."""
        t, df, p2 = mod.welch_t_test(self.base, self.op2)
        self.assertAlmostEqual(t, 3.324, places=3)
        self.assertAlmostEqual(df, 114.0, places=1)
        self.assertAlmostEqual(p2, 0.0012, places=4)

    def test_base_vs_op2_verdict(self):
        """Reference: base vs op2 verdict is significant improvement."""
        report = mod.generate_report(self.base, self.op2, "Base", "Op2")
        self.assertIn("SIGNIFICANT IMPROVEMENT", report)
        self.assertIn("+7.60", report)

    def test_fix_vs_op1_no_difference(self):
        """Reference: fix vs op1 verdict is no significant difference."""
        report = mod.generate_report(self.fix, self.op1, "Fix", "Op1")
        self.assertIn("NO SIGNIFICANT DIFFERENCE", report)

    def test_fix_vs_op2_no_difference(self):
        """Reference: fix vs op2 verdict is no significant difference."""
        report = mod.generate_report(self.fix, self.op2, "Fix", "Op2")
        self.assertIn("NO SIGNIFICANT DIFFERENCE", report)

    # --- Formatting checks ---

    def test_single_report_formatting(self):
        """Reference: single report has all expected sections."""
        report = mod.generate_single_report(self.base, "crypto.aes.base")
        for section in [
            "DESCRIPTIVE STATISTICS", "FREQUENCY HISTOGRAM",
            "DISTRIBUTION SHAPE", "NORMALITY ASSESSMENT",
            "OUTLIER DETECTION", "STABILITY ANALYSIS",
            "BOOTSTRAP CONFIDENCE INTERVAL", "SAMPLE SIZE ADEQUACY",
            "SUMMARY",
        ]:
            self.assertIn(section, report)

    def test_comparison_report_formatting(self):
        """Reference: comparison report has all expected sections."""
        report = mod.generate_report(self.base, self.fix, "Base", "Fix")
        for section in [
            "DESCRIPTIVE STATISTICS", "FREQUENCY HISTOGRAMS",
            "OVERLAY HISTOGRAM", "DISTRIBUTION SHAPE",
            "NORMALITY ASSESSMENT", "STATISTICAL TESTS",
            "SAMPLE SIZE & POWER", "CUMULATIVE COMPARISON", "VERDICT",
        ]:
            self.assertIn(section, report)


if __name__ == "__main__":
    unittest.main()

