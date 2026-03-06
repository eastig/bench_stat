# Project Structure

## Directory Layout

```
bench_stat/
├── .github/
│   └── workflows/
│       └── tests.yml            # GitHub Actions CI/CD configuration
├── .gitignore                   # Git ignore patterns
├── bench_perf.py                # Performance benchmark script
├── bench_stat.py                # Main module (~72 KB, ~2000 lines)
├── test_bench_stat.py           # Regression tests (101 tests)
├── run_tests.sh                 # Test runner script
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── PROJECT_STRUCTURE.md         # This file
├── README.md                    # Project documentation
├── TODO.md                      # Future improvements backlog
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup configuration
```

## File Descriptions

### Source Code
- **bench_stat.py**: Main module with all statistical functions and CLI entry point
- **test_bench_stat.py**: Comprehensive test suite with 101 regression tests
- **bench_perf.py**: Performance benchmark runner for key functions

### Configuration
- **setup.py**: Python package setup for pip installation
- **requirements.txt**: Project dependencies (numpy, scipy)
- **.gitignore**: Git ignore patterns for Python projects
- **.github/workflows/tests.yml**: GitHub Actions CI/CD configuration
- **run_tests.sh**: Shell script for running tests

### Documentation
- **README.md**: Project overview, features, quick start, API reference
- **CHANGELOG.md**: Version history and roadmap
- **TODO.md**: Backlog of potential library-based improvements
- **LICENSE**: MIT License

## Testing

Run tests locally:
```bash
python3 -m unittest test_bench_stat -v
```

GitHub Actions automatically runs tests on:
- Push to main/develop branches
- Pull requests
- Multiple Python versions (3.7-3.11)
- Multiple OS (Ubuntu, macOS, Windows)

## Installation

```bash
pip install git+https://github.com/eastig/bench_stat.git
```

Or clone and install:
```bash
git clone https://github.com/eastig/bench_stat.git
cd bench_stat
pip install -r requirements.txt
```
