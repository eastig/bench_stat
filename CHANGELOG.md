# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-06

### Changed

- **Bootstrap CI**: Replaced custom percentile bootstrap with `scipy.stats.bootstrap`
  using BCa (bias-corrected and accelerated) method. BCa provides better coverage
  for skewed distributions.
- **Permutation test**: Replaced custom vectorized batching with
  `scipy.stats.permutation_test`. Supports exact permutations for small samples.
- **Welch's t-test**: Uses `scipy.stats.ttest_ind().df` instead of manual
  Welch-Satterthwaite degrees-of-freedom calculation.
- **RNG**: Bootstrap and permutation functions use `np.random.default_rng` (modern
  numpy Generator API) instead of deprecated `np.random.RandomState`.
- **Minimum requirements**: Python >=3.9, numpy >=1.24, scipy >=1.11.
- **Warnings**: `load_csv` uses `stacklevel=2` for better warning messages.

### Removed

- `_ci_index_bounds` helper (no longer needed with scipy.stats.bootstrap).

### Testing

- 139 regression tests (up from 101).
- Added reference data tests against `test_data/` CSV inputs.
- Added correctness tests for all statistical functions against scipy reference.

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
