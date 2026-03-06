# bench_stat - Statistical Benchmarking Analysis Tool

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-101/101-brightgreen.svg)]()

A comprehensive statistical analysis tool for performance benchmarks with confidence intervals, statistical significance testing, and cumulative analysis.

## Features

- **Descriptive Statistics**: Mean, standard deviation, quartiles, skewness, kurtosis
- **Confidence Intervals**: Bootstrap and parametric (Student's t) CIs
- **Statistical Tests**: Welch's t-test, Mann-Whitney U, permutation test, bootstrap test
- **Effect Size**: Cohen's d calculation with interpretation
- **Power Analysis**: Statistical power, minimum detectable effect, required sample size
- **Data Quality**: Normality testing (D'Agostino-Pearson), outlier detection (IQR and z-score)
- **Cumulative Analysis**: Threshold-based comparison tables with optimized O(n log n) complexity
- **Comprehensive Reporting**: Single dataset analysis and comparison reports
- **Visualization**: ASCII histograms, stability charts, distribution shapes, p-value visual scales

## Installation

### From Source

```bash
git clone https://github.com/eastig/bench_stat.git
cd bench_stat
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all regression tests
python3 -m unittest test_bench_stat -v

# Run specific test class
python3 -m unittest test_bench_stat.BenchStatRegressionTests -v

# Run performance benchmarks
python3 bench_perf.py
```

## Quick Start

### Single Dataset Analysis

```python
import bench_stat as mod

data = [100.5, 101.2, 99.8, 102.1, 100.9, 98.7, 101.5, 99.2]
report = mod.generate_single_report(data, 'My Benchmark', alpha=0.05)
print(report)
```

### Comparison Analysis

```python
import bench_stat as mod

baseline = [100, 101, 99, 100, 102, 98, 101, 100]
experiment = [105, 106, 104, 105, 107, 103, 106, 105]

report = mod.generate_report(baseline, experiment, 'Baseline', 'Experiment', alpha=0.05)
print(report)
```

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

## Project Structure

```
bench_stat/
├── .github/workflows/tests.yml  # GitHub Actions CI/CD
├── bench_stat.py                # Main module (~2000 lines)
├── test_bench_stat.py           # Regression tests (101 tests)
├── bench_perf.py                # Performance benchmark script
├── setup.py                     # Package setup file
├── requirements.txt             # Python dependencies
├── run_tests.sh                 # Test runner script
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── TODO.md                      # Future improvements backlog
├── PROJECT_STRUCTURE.md         # Detailed structure documentation
└── LICENSE                      # MIT License
```

## Key Functions

### generate_single_report(data, label, higher_is_better=True, alpha=0.05)
Generates comprehensive statistical report for a single dataset.

**Returns**: Formatted string with:
- Descriptive statistics (n, mean, std dev, quartiles, skewness, kurtosis)
- Bootstrap and parametric confidence intervals
- CI agreement assessment
- Sample size adequacy analysis
- Normality testing
- Outlier detection
- Stability chart

### generate_report(data_base, data_exp, label_base, label_exp, higher_is_better=True, alpha=0.05)
Generates comparison report for two datasets.

**Returns**: Formatted string with:
- Descriptive statistics for both datasets
- Statistical tests (Welch's t-test, Mann-Whitney U, permutation, bootstrap)
- Effect size (Cohen's d)
- Cumulative threshold comparison table
- Power analysis and sample size recommendations
- P-value visual scales

### cumulative_comparison(data1, data2, label1, label2)
Generates cumulative threshold comparison table showing percentage of values ≥ threshold.

**Features**:
- Dynamic threshold selection based on data distribution
- Percentage comparison with difference highlighting
- "Better" indicator (>2% difference threshold)
- Vectorized np.searchsorted optimization for large datasets

## API Reference

### Data Loading

- `load_csv(filename)` - Load numeric data from a CSV file

### Descriptive Statistics

- `compute_descriptive_stats(data)` - Compute all descriptive statistics (n, mean, var, stdev, se, median, min, max, range, quartiles, skewness, kurtosis)

### Confidence Intervals

- `bootstrap_ci(data1, data2, n_boot=10000, ci=95, seed=42)` - Bootstrap CI for difference in means
- `bootstrap_single_ci(data, n_boot=10000, ci=95, seed=42)` - Bootstrap CI for a single dataset mean
- `_ci_index_bounds(sample_count, ci)` - CI index calculation for sorted bootstrap samples

### Statistical Tests

- `welch_t_test(data1, data2)` - Welch's t-test (unequal variances)
- `mann_whitney_u(data1, data2)` - Mann-Whitney U test (non-parametric, with tie correction)
- `permutation_test(data1, data2, n_perms=10000, seed=42)` - Permutation test for difference in means
- `dagostino_pearson_test(data)` - D'Agostino-Pearson normality test

### Effect Size and Power

- `statistical_power(n1, n2, d, alpha=0.05)` - Power via noncentral t-distribution
- `minimum_detectable_effect(n1, n2, alpha=0.05, target_power=0.80)` - MDE at given power
- `required_sample_size(d, alpha=0.05, target_power=0.80)` - Required n per group

### Outlier Detection

- `detect_outliers_iqr(data, q1, q3, iqr)` - IQR-based outlier detection
- `detect_outliers_zscore(data, m, s, threshold=3.0)` - Z-score-based outlier detection

### Report Generation

- `generate_single_report(data, label, higher_is_better=True, alpha=0.05)` - Single dataset report
- `generate_report(data_base, data_exp, label_base, label_exp, higher_is_better=True, alpha=0.05)` - Comparison report

## Performance Optimizations

- O(n log n) cumulative comparison via vectorized `np.searchsorted`
- Vectorized batched permutation test (replaces Python loop)
- Vectorized bootstrap CI with memory-efficient batching
- Mann-Whitney U with scipy tie correction (replaces manual ranking)
- Unbiased skewness/kurtosis derived from `scipy.stats.describe` (avoids extra array passes)
- Single array conversion in `welch_t_test` (eliminates redundant list-to-array conversions)
- Precomputed stats reuse across report sections

## Testing

The project includes 101 comprehensive regression tests covering:
- Statistical correctness (normality, statistical tests, effect sizes)
- Edge cases (n=1, zero values, identical datasets, zero variance)
- Infinity handling in confidence intervals
- Display consistency and formatting
- Performance optimizations
- Bootstrap batching

```bash
python3 -m unittest test_bench_stat -v
# Output: Ran 101 tests in ~8 seconds ... OK
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use bench_stat in your research or benchmarking work, please cite:

```bibtex
@software{bench_stat_2026,
  title = {bench_stat: Statistical Benchmarking Analysis Tool},
  author = {Eastig},
  year = {2026},
  url = {https://github.com/eastig/bench_stat}
}
```
