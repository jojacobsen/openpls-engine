# Changelog

All notable changes to `openpls-engine` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0 releases live on the `0.x` line while the API stabilizes. Tagged
releases (`vX.Y.Z`) trigger a GitHub Actions workflow that builds the
package and publishes it to PyPI via OIDC trusted publishing.

## [Unreleased]

## [0.7.0a2] - 2026-05-30

First release published to PyPI. Identical code to `0.7.0a1`; bumped only
to validate the trusted-publisher pipeline end to end. The previous
`v0.7.0a1` GitHub release stays available as a download but was never
uploaded to PyPI.

## [0.7.0a1] - 2026-05-30

All planned ports from the OpenPLS web app are now in. This is the first
feature-complete pre-release.

### Added
- `plspm.long_bootstrap.LongBootstrap`: serial bootstrap with progress
  callback, sign-flipping, BCa percentile CIs, normal-approximation
  p-values, and a configurable success-rate floor. Suited for long-running,
  progress-streaming workloads.
- `plspm.mga.MGA` and `plspm.mga.GroupSpec`: Multi-Group Analysis via
  Henseler permutation, with categorical and numeric-range group
  definitions, pairwise comparisons across 2+ groups, two-sided permutation
  p-values with Phipson-Smyth add-one smoothing.
- `Plspm(..., missing_strategy="mean")`: mean replacement for NaN cells in
  indicator columns. Default `"casewise"` preserves upstream behavior.
- `plspm.q_squared.QSquared`: Stone-Geisser Q² via blindfolding with
  configurable omission distance D. Exposed as `Plspm.q_squared()`.
- `plspm.htmt.HTMT`: Heterotrait-Monotrait ratio of correlations.
- `plspm.fit.ModelFit`: SRMR (Standardized Root Mean Square Residual) and
  d_ULS (unweighted least-squares discrepancy).
- BIC for endogenous LVs in `plspm.inner_summary`.
- Listwise-deletion fallback for Cronbach α and Dijkstra-Henseler ρ when an
  LV's indicator block contains NaN.
- `plspm.__version__` reports the installed package version at runtime.

### Changed
- Project metadata moved from `setup.py` to PEP 621 `pyproject.toml`.
- Lint pipeline (ruff) and test pipeline (pytest) run on Python 3.10 through
  3.13 in CI.

## [0.6.0a1] - 2026-05-30

Initial OpenPLS rebrand of the `plspm-python` 0.5.7 baseline.

### Added
- Forked `plspm-python` 0.5.7 with attribution preserved.
- `pyproject.toml`, ruff config, GitHub Actions CI matrix (Py 3.10 to 3.13).

[Unreleased]: https://github.com/jojacobsen/openpls-engine/compare/v0.7.0a2...HEAD
[0.7.0a2]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a2
[0.7.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a1
[0.6.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.6.0a1
