#!/bin/bash
# Run regression tests for bench_stat.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running bench_stat regression tests..."
python3 test_bench_stat.py "$@"
echo "All tests passed!"
