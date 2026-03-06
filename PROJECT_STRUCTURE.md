# Project Structure

## Directory Layout

```
bench_stat/
├── .github/
│   └── workflows/
│       └── tests.yml            # GitHub Actions CI/CD (Python 3.9-3.13)
├── .gitignore                   # Git ignore patterns
├── bench_perf.py                # Performance benchmarks
├── bench_stat.py                # Main module
├── test_bench_stat.py           # Regression tests (137 tests)
├── test_data/                   # CSV inputs and reference outputs
│   ├── crypto.aes.base.csv
│   ├── crypto.aes.fix.csv
│   ├── crypto.aes.fix.op1.csv
│   ├── crypto.aes.fix.op2.csv
│   ├── base_stats.txt           # Reference single report
│   ├── base_vs_fix.txt          # Reference comparison report
│   └── ...                      # Other reference outputs
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── PROJECT_STRUCTURE.md         # This file
├── README.md                    # Project documentation
├── TODO.md
├── requirements.txt             # Dependencies (numpy >=1.24, scipy >=1.11)
└── setup.py                     # Package setup (Python >=3.9)
```

## File Descriptions

### Source Code
- **bench_stat.py**: Main module with all statistical functions and CLI entry point
- **test_bench_stat.py**: 137 regression tests with reference data validation
- **bench_perf.py**: Performance benchmark runner

### Configuration
- **setup.py**: Python package setup for pip installation
- **requirements.txt**: Production dependencies
- **.github/workflows/tests.yml**: CI/CD across Python 3.9-3.13, Ubuntu/macOS/Windows

### Test Data
- **test_data/*.csv**: Input benchmark data for reference tests
- **test_data/*.txt**: Reference outputs for regression validation
- **test_data/benchmark_baseline.txt**: Performance baseline

### Documentation
- **README.md**: Overview, quick start, API reference
- **CHANGELOG.md**: Version history
- **LICENSE**: MIT License

## Testing

```bash
python3 -m unittest test_bench_stat -v
```

GitHub Actions runs on push/PR against Python 3.9-3.13 on Ubuntu, macOS, Windows.

## Installation

```bash
pip install git+https://github.com/eastig/bench_stat.git
```

For Python 3.7: use the `python_3.7.16` branch.
