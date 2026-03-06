# bench_stat - Statistical Benchmarking Analysis Tool

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-137/137-brightgreen.svg)]()

A statistical analysis tool for performance benchmarks with confidence intervals, statistical significance testing, and cumulative analysis.

Requires at least 3 data points per dataset.

## Features

- **Descriptive Statistics**: Mean, standard deviation, quartiles, skewness, kurtosis
- **Confidence Intervals**: BCa bootstrap and parametric (Student's t) CIs
- **Statistical Tests**: Welch's t-test, Mann-Whitney U, permutation test, bootstrap test
- **Effect Size**: Cohen's d calculation with interpretation
- **Power Analysis**: Statistical power, minimum detectable effect, required sample size
- **Data Quality**: Normality testing (D'Agostino-Pearson), outlier detection (IQR and z-score)
- **Cumulative Analysis**: Threshold-based comparison tables
- **Comprehensive Reporting**: Single dataset analysis and comparison reports
- **Visualization**: ASCII histograms, stability charts, distribution shapes, p-value visual scales

## Installation

```bash
git clone https://github.com/eastig/bench_stat.git
cd bench_stat
pip install -r requirements.txt
```

For Python 3.7 compatibility, use the `python_3.7.16` branch (numpy ≥1.15, scipy ≥1.7).

## Quick Start

### Command-Line Usage

```bash
# Single experiment analysis
python3 bench_stat.py results.csv

# Compare two experiments (higher values = better)
python3 bench_stat.py baseline.csv experiment.csv

# Compare two experiments (lower values = better, e.g. latency)
python3 bench_stat.py base.csv fix.csv --lower-is-better

# Custom significance level and labels
python3 bench_stat.py base.csv fix.csv --alpha 0.01 --base-label "Baseline" --exp-label "Fix"
```

### Python API

```python
import bench_stat as mod

# Single dataset
data = [100.5, 101.2, 99.8, 102.1, 100.9, 98.7, 101.5, 99.2]
report = mod.generate_single_report(data, 'My Benchmark', alpha=0.05)
print(report)

# Comparison
baseline = [100, 101, 99, 100, 102, 98, 101, 100]
experiment = [105, 106, 104, 105, 107, 103, 106, 105]
report = mod.generate_report(baseline, experiment, 'Baseline', 'Experiment', alpha=0.05)
print(report)
```

## Example Output

### Single Dataset Analysis

```
========================================================================
  BENCHMARK ANALYSIS REPORT
  baseline.data
  Higher is better
========================================================================

────────────────────────────────────────────────────────────────────────
  DESCRIPTIVE STATISTICS
────────────────────────────────────────────────────────────────────────
  Metric                               Value
  ────────────────────  ────────────────────
  n                                       58
  Mean                               1054.53
  Std Dev                              12.24
  Std Error                             1.61
  Median                             1054.37
  Min                                1030.72
  Max                                1089.65
  Range                                58.93
  25th pct (Q1)                      1045.79
  75th pct (Q3)                      1059.70
  IQR                                  13.91
  Skewness                             0.633
  Ex. Kurtosis                         0.712
  95% CI (mean)           [1051.31, 1057.75]

  Coefficient of Variation: 1.16%
    → Low variability — reasonably consistent results

────────────────────────────────────────────────────────────────────────
  FREQUENCY HISTOGRAM
────────────────────────────────────────────────────────────────────────
  baseline.data
  ops/min           Frequency
  [1030.7,1037.3) │ ████████████                               4
  [1037.3,1043.8) │ █████████████████████                      7
  [1043.8,1050.4) │ █████████████████████████████████         11
  [1050.4,1056.9) │ ████████████████████████████████████      12  ← mean (1054.53)
  [1056.9,1063.5) │ ████████████████████████████████████████  13  ← mode
  [1063.5,1070.0) │ ███████████████                            5
  [1070.0,1076.6) │ █████████                                  3
  [1076.6,1083.1) │ ███                                        1
  [1083.1,1089.7) │ ██████                                     2
                  └─────────────────────────────────────────────
                    Mean = 1054.53  Std = 12.24  n = 58

  ... (normality assessment, outlier detection, stability chart) ...

────────────────────────────────────────────────────────────────────────
  SUMMARY
────────────────────────────────────────────────────────────────────────
  Benchmark: baseline.data
  Result:    1054.53 ± 12.24 ops/min (mean ± std)
  95% CI:    [1051.31, 1057.75] ops/min
  CV:        1.16%
  Outliers:  2 (IQR method)
  Normal:    Yes (p = 0.0667)

  Recommendations:
  ⚠️  2 outlier(s) detected — consider investigating whether they
      represent real behavior or measurement errors

========================================================================
```

### Two-Dataset Comparison

```
========================================================================
  BENCHMARK COMPARISON REPORT
  baseline.data (baseline) vs. fix.data (experiment)
  Higher is better
========================================================================

────────────────────────────────────────────────────────────────────────
  DESCRIPTIVE STATISTICS
────────────────────────────────────────────────────────────────────────
  Metric           baseline.data   fix.data
  ───────────────  ───────────────  ───────────────
  n                             58               58
  Mean                     1054.53          1059.45
  Median                   1054.37          1059.66
  Std Dev                    12.24            11.96
  Min                      1030.72          1028.77
  Max                      1089.65          1090.31
  25th pct                 1045.79          1051.51
  75th pct                 1059.70          1068.27
  Skewness                   0.633            0.047
  Ex. Kurtosis               0.712            0.094

  Observed difference: +4.92 (+0.47%)
  Direction: ✅ Improvement

  ... (histograms, normality assessment) ...

────────────────────────────────────────────────────────────────────────
  STATISTICAL TESTS
────────────────────────────────────────────────────────────────────────
  Hypothesis: fix.data > baseline.data
  Significance level: α = 0.05

  Test                  Statistic                  p (1-tail)  p (2-tail)        Significant?
  ────────────────────  ─────────────────────────  ──────────  ──────────  ──────────────────
  Welch's t-test        t=2.191, df=113.9              0.0152      0.0305               ✅ Yes
  Mann-Whitney U        U=1254, z=2.363                0.0091      0.0181               ✅ Yes
  Permutation test      δ=4.92                         0.0156      0.0312               ✅ Yes
  Bootstrap             CI=[0.47, 9.18]                0.0132      0.0264               ✅ Yes

  Cohen's d: 0.407 (Small)
    Interpretation: The means are 0.41 pooled standard deviations apart.
    Win probability: A random fix.data value beats a random
    baseline.data value ~61% of the time.

  p-value visual summary (one-tailed):

  Test                  p-value  0.00  0.01  0.02  0.03  0.04  0.05
                                 │     │     │     │     │     │
  Welch's t-test         0.0152  │         ▓
  Mann-Whitney U         0.0091  │     ▓
  Permutation test       0.0156  │         ▓
  Bootstrap              0.0132  │       ▓
                                 │                              α=0.0500

  ... (power analysis) ...

────────────────────────────────────────────────────────────────────────
  CUMULATIVE COMPARISON
────────────────────────────────────────────────────────────────────────
  Percentage of values ≥ threshold:

   Threshold  baseline.data   fix.data  Difference           Better
  ──────────  ───────────────  ───────────────  ──────────  ───────────────
      ≥ 1030             100%              98%         -2%           ≈ tie
      ≥ 1036              95%              98%         +3%  fix.data
      ≥ 1042              86%              97%        +10%  fix.data
      ≥ 1048              69%              79%        +10%  fix.data
      ≥ 1055              47%              62%        +16%  fix.data
      ≥ 1061              22%              43%        +21%  fix.data
      ≥ 1067              14%              31%        +17%  fix.data
      ≥ 1073               9%              12%         +3%  fix.data
      ≥ 1079               3%               3%         +0%           ≈ tie

────────────────────────────────────────────────────────────────────────
  VERDICT
────────────────────────────────────────────────────────────────────────
  Tests significant at α=0.05: 4/4
  Mean difference: +4.92 (+0.47%)
  Cohen's d: 0.407 (Small)
  Bootstrap 95% CI: [0.47, 9.18]

  ✅ SIGNIFICANT IMPROVEMENT: fix.data is significantly better
     than baseline.data.

  Practical significance: SMALL — real but modest improvement.
     Consider code quality and other factors.

========================================================================
```

## Project Structure

```
bench_stat/
├── .github/workflows/tests.yml  # GitHub Actions CI/CD
├── bench_stat.py                # Main module
├── test_bench_stat.py           # Regression tests (137 tests)
├── bench_perf.py                # Performance benchmarks
├── test_data/                   # CSV inputs and reference outputs
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies (numpy >=1.24, scipy >=1.11)
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
└── TODO.md
```

## API Reference

### Data Loading

- `load_csv(filename)` - Load numeric data from a CSV file

### Descriptive Statistics

- `compute_descriptive_stats(data)` - All descriptive statistics (requires n ≥ 3)

### Confidence Intervals

- `bootstrap_ci(data1, data2, n_boot, ci, seed)` - BCa bootstrap CI for difference in means
- `bootstrap_single_ci(data, n_boot, ci, seed)` - BCa bootstrap CI for single dataset mean

### Statistical Tests

- `welch_t_test(data1, data2)` - Welch's t-test via scipy.stats.ttest_ind
- `mann_whitney_u(data1, data2)` - Mann-Whitney U test with tie correction
- `permutation_test(data1, data2, n_perms, seed)` - via scipy.stats.permutation_test
- `dagostino_pearson_test(data)` - D'Agostino-Pearson normality test

### Effect Size and Power

- `statistical_power(n1, n2, d, alpha)` - Power via noncentral t-distribution
- `minimum_detectable_effect(n1, n2, alpha, target_power)` - MDE at given power
- `required_sample_size(d, alpha, target_power)` - Required n per group

### Outlier Detection

- `detect_outliers_iqr(data, q1, q3, iqr)` - IQR-based outlier detection
- `detect_outliers_zscore(data, m, s, threshold)` - Z-score-based outlier detection

### Report Generation

- `generate_single_report(data, label, higher_is_better, alpha)` - Single dataset report
- `generate_report(data_base, data_exp, label_base, label_exp, higher_is_better, alpha)` - Comparison report

## Testing

```bash
python3 -m unittest test_bench_stat -v
# Ran 137 tests in ~8 seconds ... OK

python3 bench_perf.py  # Performance benchmarks
```

137 tests covering statistical correctness against scipy reference implementations,
edge cases, formatting, and reference data regression tests against `test_data/` CSVs.
Unexpected warnings are treated as test failures.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{bench_stat_2026,
  title = {bench_stat: Statistical Benchmarking Analysis Tool},
  author = {Eastig},
  year = {2026},
  url = {https://github.com/eastig/bench_stat}
}
```
