#!/usr/bin/env python3
"""Performance benchmarks for bench_stat.py.

Measures key function timings across dataset sizes to track performance.

Usage:
    python3 bench_perf.py
"""

import time
import numpy as np
import bench_stat as mod


def _generate_data(n1, n2, seed=42, shift=2.0):
    """Generate two normally distributed datasets for benchmarking."""
    rng = np.random.RandomState(seed)
    data1 = rng.normal(100, 15, n1).tolist()
    data2 = rng.normal(100 + shift, 15, n2).tolist()
    return data1, data2


def _print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _time_call(func, *args, **kwargs):
    """Time a single function call, return (elapsed_ms, result)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, result


def benchmark_cumulative_comparison():
    """Benchmark cumulative_comparison across dataset sizes."""
    _print_header("cumulative_comparison()")

    sizes = [("Small", 100), ("Medium", 1000), ("Large", 5000), ("XLarge", 10000)]

    print(f"\n{'Label':<10} {'n':>8} {'Time (ms)':>12} {'Thresholds':>12}")
    print("-" * 44)

    for label, n in sizes:
        data1, data2 = _generate_data(n, n)
        elapsed, result = _time_call(mod.cumulative_comparison, data1, data2, "Base", "Exp")
        n_thresh = sum(1 for line in result.split('\n') if '≥' in line)
        print(f"{label:<10} {n:>8} {elapsed:>12.3f} {n_thresh:>12}")


def benchmark_permutation_test():
    """Benchmark permutation_test across dataset sizes."""
    _print_header("permutation_test()")

    cases = [("Tiny", 10), ("Small", 50), ("Medium", 200), ("Large", 1000)]
    n_perms = 10000

    print(f"\n{'Label':<10} {'n':>6} {'n_perms':>10} {'Time (ms)':>12} {'p-value':>10}")
    print("-" * 52)

    for label, n in cases:
        data1, data2 = _generate_data(n, n)
        elapsed, (diff, p) = _time_call(mod.permutation_test, data1, data2, n_perms=n_perms, seed=42)
        print(f"{label:<10} {n:>6} {n_perms:>10} {elapsed:>12.3f} {p:>10.4f}")


def benchmark_bootstrap_functions():
    """Benchmark bootstrap_ci and bootstrap_single_ci across dataset sizes."""
    _print_header("bootstrap_ci() & bootstrap_single_ci()")

    cases = [("Tiny", 10), ("Small", 50), ("Medium", 200), ("Large", 500)]
    n_boot = 10000

    print(f"\n  bootstrap_ci:")
    print(f"  {'Label':<10} {'n':>6} {'n_boot':>10} {'Time (ms)':>12} {'CI Width':>10}")
    print("  " + "-" * 52)

    for label, n in cases:
        data1, data2 = _generate_data(n, n)
        elapsed, (ci_lo, ci_hi, _) = _time_call(mod.bootstrap_ci, data1, data2, n_boot=n_boot, seed=42)
        print(f"  {label:<10} {n:>6} {n_boot:>10} {elapsed:>12.3f} {ci_hi - ci_lo:>10.2f}")

    print(f"\n  bootstrap_single_ci:")
    print(f"  {'Label':<10} {'n':>6} {'n_boot':>10} {'Time (ms)':>12} {'CI Width':>10}")
    print("  " + "-" * 52)

    for label, n in cases:
        data1, _ = _generate_data(n, 1)
        elapsed, (ci_lo, ci_hi) = _time_call(mod.bootstrap_single_ci, data1, n_boot=n_boot, seed=42)
        print(f"  {label:<10} {n:>6} {n_boot:>10} {elapsed:>12.3f} {ci_hi - ci_lo:>10.2f}")


def benchmark_mann_whitney_u():
    """Benchmark mann_whitney_u across dataset sizes."""
    _print_header("mann_whitney_u()")

    cases = [("Tiny", 20), ("Small", 50), ("Medium", 200), ("Large", 500)]

    print(f"\n{'Label':<10} {'n':>6} {'Time (ms)':>12} {'U-stat':>12} {'z-stat':>10}")
    print("-" * 52)

    for label, n in cases:
        data1, data2 = _generate_data(n, n)
        elapsed, (u, z, _) = _time_call(mod.mann_whitney_u, data1, data2)
        print(f"{label:<10} {n:>6} {elapsed:>12.3f} {u:>12.1f} {z:>10.3f}")


def benchmark_generate_single_report():
    """Benchmark generate_single_report across dataset sizes."""
    _print_header("generate_single_report()")

    cases = [
        ("Tiny (n=1)", [5.0]),
        ("Small (n=10)", list(range(1, 11))),
        ("Medium (n=100)", list(range(1, 101))),
        ("Large (n=1000)", list(range(1, 1001))),
    ]

    print(f"\n{'Label':<15} {'n':>6} {'Time (ms)':>12}")
    print("-" * 35)

    for label, data in cases:
        elapsed, _ = _time_call(mod.generate_single_report, data, label, alpha=0.05)
        print(f"{label:<15} {len(data):>6} {elapsed:>12.3f}")


def benchmark_generate_report():
    """Benchmark generate_report across dataset sizes."""
    _print_header("generate_report()")

    cases = [
        ("Small", list(range(10, 20)), list(range(11, 21))),
        ("Medium", list(range(100)), list(range(1, 101))),
        ("Large", list(range(1000)), list(range(1, 1001))),
    ]

    print(f"\n{'Label':<10} {'n1':>6} {'n2':>6} {'Time (ms)':>12}")
    print("-" * 37)

    for label, base, exp in cases:
        elapsed, _ = _time_call(mod.generate_report, base, exp, "Base", "Exp", alpha=0.05)
        print(f"{label:<10} {len(base):>6} {len(exp):>6} {elapsed:>12.3f}")


def main():
    print()
    print("#" * 70)
    print("# BENCH_STAT PERFORMANCE BENCHMARKS")
    print("#" * 70)

    benchmark_cumulative_comparison()
    benchmark_permutation_test()
    benchmark_bootstrap_functions()
    benchmark_mann_whitney_u()
    benchmark_generate_single_report()
    benchmark_generate_report()

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
