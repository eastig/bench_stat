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
