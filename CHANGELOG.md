# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-05

### Added

- Initial release of bench_stat
- Complete statistical analysis framework for benchmarking
- **Features**:
  - Descriptive statistics (mean, std dev, quartiles, skewness, kurtosis)
  - Bootstrap confidence intervals with configurable size
  - Parametric (Student's t) confidence intervals
  - Multiple statistical tests:
    - Welch's t-test (assumes unequal variances)
    - Mann-Whitney U test (non-parametric, with tie correction)
    - Permutation test (vectorized batched)
    - Bootstrap test
  - Effect size computation (Cohen's d)
  - Power analysis (noncentral t-distribution)
  - Data quality analysis:
    - Normality testing (D'Agostino-Pearson)
    - Outlier detection (IQR and z-score methods)
  - Cumulative threshold analysis
  - ASCII histograms, stability charts, distribution shape comparison
  - P-value visual scales with dynamic adaptation
  - Comprehensive reporting for single datasets and comparisons
  - Command-line interface with argparse (single and comparison modes)
  - CSV data loading via numpy genfromtxt

- **Testing**:
  - 101 comprehensive regression tests
  - Tests for correctness, edge cases, and performance
  - CI/CD with GitHub Actions
  - Multi-platform testing (Ubuntu, macOS, Windows)

- **Project Setup**:
  - setup.py for pip installation
  - requirements.txt for dependency management
  - GitHub Actions CI/CD workflow
