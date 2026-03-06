# TODO

## Second Pass: Library-Based Improvements (No Code Changes Yet)

This backlog captures possible refactors to replace custom logic with standard
NumPy/SciPy APIs where practical.

## High Priority

- [ ] Add optional SciPy-native permutation test path in `bench_stat.py:818`
  - Candidate API: `scipy.stats.permutation_test`.
  - Why: Replace custom permutation sampling/shuffling loop with a maintained
    library implementation.
  - Constraint: Not available in current minimum supported SciPy (`>=1.7.0`).
  - Suggested approach: Feature-detect at runtime and fallback to current
    implementation when unavailable.

- [ ] Add optional SciPy-native Welch df extraction in `bench_stat.py:723`
  - Candidate API: `scipy.stats.ttest_ind(...).df`.
  - Why: Remove manual Welch-Satterthwaite degrees-of-freedom calculation.
  - Constraint: `Ttest_indResult.df` is not present in SciPy 1.7.x.
  - Suggested approach: Use result `df` only when attribute exists;
    otherwise keep current manual formula.

## Medium Priority

- [ ] Consider replacing `_t_critical` guard logic in `bench_stat.py:712`
  - Candidate APIs: `scipy.stats.norm.isf` or direct `ppf` with explicit clipping.
  - Why: Could simplify edge-case handling and make tail intent clearer.
  - Note: Current implementation is already robust and fast; this is mostly a
    readability/consistency cleanup.

- [ ] Consider replacing CI index math helper `_ci_index_bounds` in
      `bench_stat.py:930`
  - Candidate API: `numpy.quantile` (with explicit interpolation/method policy).
  - Why: Express percentile bounds using a standard quantile interface.
  - Risk: May change exact index semantics vs current integer-index behavior,
    so requires strict regression tests before adoption.

## Low Priority

- [ ] Investigate SciPy power-analysis helper usage for required sample size
      in `bench_stat.py:1002`
  - Candidate APIs: `scipy.stats.nct` + higher-level wrappers if adopted in
    future dependency versions.
  - Why: Potentially reduce custom glue in power target solving.
  - Note: Current `scipy.optimize`-based implementation is already library-based
    and validated; this is only future simplification.

## Optional Dependency Policy Follow-Up

- [ ] Evaluate raising minimum SciPy version from `>=1.7.0` once compatibility
      matrix allows it.
  - Benefit: Enables cleaner use of newer stats APIs (`permutation_test`, richer
    test result objects).
  - Impact: Packaging and CI matrix updates required.

## Test/Validation Additions (When Implementing Any Item)

- [ ] Add version-gated tests ensuring fallback behavior remains correct when
      newer SciPy APIs are absent.
- [ ] Add parity tests comparing fallback vs native-library paths on fixed seeds
      and representative datasets.
- [ ] Add performance smoke tests to ensure no regressions in large-n scenarios.
