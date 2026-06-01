# Changelog

All notable changes to `openpls-engine` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0 releases live on the `0.x` line while the API stabilizes. Tagged
releases (`vX.Y.Z`) trigger a GitHub Actions workflow that builds the
package and publishes it to PyPI via OIDC trusted publishing.

## [Unreleased]

## [1.0.1] - 2026-06-01

Two SmartPLS-parity fixes discovered while validating 1.0.0 against 14
reference cases. No API changes.

### Fixed
- **Per-LV sign vote in `_MetricWeights.calculate()`**: applied
  `math.copysign(1.0, x)` to the masked product `cor * odm` instead of
  computing the sign first and masking after. Because `copysign(1.0, 0)`
  returns `+1.0`, every non-belonging cell contributed a phantom `+1`
  to the LV's vote, so a small LV (e.g. 3 indicators) embedded in a much
  larger model could be out-voted by the large LV's phantom contributions —
  leaving the small LV on the wrong sign even when every one of its
  indicators correlated negatively with the latent direction. Now the
  sign is computed first and then multiplied by the membership mask, so
  non-belonging cells contribute `0` rather than `+1`. Empirical case:
  the OI / OI-variations validation cases now match SmartPLS on
  `Org_Ident → AC_Love` (β was `+0.41` vs SmartPLS `−0.41`).
- **Saturated-model SRMR / d_ULS excludes within-LV pairs for Mode B
  (formative) constructs**: the implied indicator-correlation matrix
  `Σ̂ = Λ Φ Λᵀ` only constrains common-factor (Mode A) measurement.
  Mode B indicators are exogenous causes of the composite, so their
  pairwise correlation is empirical, not implied by `Λᵢ Λⱼ`. Including
  those pairs in the SRMR / d_ULS sums inflated both metrics purely as a
  measurement-model artifact (Henseler et al. 2014 §5.3, SmartPLS
  convention). The fit now builds an inclusion mask that excludes
  within-Mode-B-LV blocks and aggregates over the kept pairs only.
  Models without Mode B LVs are unaffected. Empirical case: the
  Corporate Reputation Advanced validation case d_ULS gap closes from
  `+0.5104` to `-0.0004`.

### Added
- Regression test `tests/test_sign_convention.py` constructing a
  14-indicator LV next to a 3-indicator LV whose indicators are all
  inverted; pins the sign-vote behaviour against the old phantom-vote bug.
- Regression test `tests/test_fit.py::test_mode_b_within_lv_pairs_excluded_from_fit`
  asserting that the within-Mode-B residual block is masked out of both
  SRMR and d_ULS sums by exact arithmetic identity.

## [1.0.0] - 2026-06-01

First stable release. Two breaking changes plus a numerical fix that brings
mixed-scale models in line with SmartPLS 4. The public algorithm surface
(Plspm, Config, Mode, Scheme, IPMA, PLSpredict, Moderation, FIMIX) is now
considered stable under semver.

### Breaking
- **Import namespace renamed from `plspm` to `openpls`.** Replace every
  `from plspm.<sub> import …` / `import plspm.<sub> as …` with the
  `openpls.<sub>` equivalent. `Plspm` is now re-exported at the top level,
  so the recommended entry point becomes `from openpls import Plspm`.
  The PyPI distribution name (`openpls-engine`) is unchanged.
- The internal module `openpls.plspm` keeps the class `Plspm` (the
  algorithm name), but consumers should prefer the top-level import.

### Fixed
- `openpls.config.Config.treat_data()` now standardizes each indicator
  column with its own Bessel-corrected (ddof=1) sample standard deviation,
  matching SmartPLS 4. The upstream behaviour pooled the standard
  deviation across all indicators in a latent variable's block, which
  caused the high-variance indicator to dominate composite scores when
  indicators had different native scales (e.g. Likert 1–7 mixed with a
  percentile 0–100 measure). Loadings, R², HTMT, f² and Q² are
  scale-invariant and therefore unaffected; outer weights and composite
  scores now align with the SmartPLS convention. (Empirical case:
  Employee Retention `JS` latent variable now reports JS1–JS5 loadings of
  0.78, 0.80, 0.76, 0.76, 0.82 — within ±0.0005 of SmartPLS — instead of
  the previous collapse onto JS5.)
- `openpls.util.treat()` accepts a pandas `Series` of column-wise scale
  values without the `if scale_values:` truthy check raising
  `"The truth value of a Series is ambiguous"`.
- `openpls.config.Config.add_lv()` no longer rejects manifest variables
  whose name matches a latent variable. ECSI-style models with single-item
  LVs (e.g. CUSCO) previously crashed at config time; MV and LV names live
  in separate internal namespaces, so the check was defensive only.

### Removed
- Legacy `setup.py` shim. The build is driven entirely by
  `pyproject.toml` (PEP 621); `pip install` and `python -m build` behave
  identically.

## [0.7.0a4] - 2026-05-30

Metadata-only patch. Adds a `Documentation` link to PyPI's project sidebar
pointing at the new docs site at https://openpls.app/engine/. No code changes.

### Changed
- `pyproject.toml`: add `Documentation = "https://openpls.app/engine/"` to
  `[project.urls]` so the PyPI project page surfaces the docs alongside
  Homepage, Issues and the OpenPLS web app.

## [0.7.0a3] - 2026-05-30

Second feature release. Ships four advanced PLS-SEM analyses (IPMA,
PLSpredict, two-stage moderation, FIMIX-PLS) and two additional inner-
weighting schemes (Newton/BFGS and Lohmöller's PCA), filling the gap
between the original `plspm-python` API and mainstream commercial PLS-SEM
tools.

### Added
- `Scheme.PCA`: Lohmöller's PCA inner-weighting scheme (Lohmöller 1989,
  §2.4.2). For each LV, the inner weights are the components of the first
  principal direction of its neighbor-score matrix, sign-flipped to
  correlate positively with the LV. Treats neighbor weights as a joint
  multivariate direction rather than as pairwise quantities.
- `Scheme.NEWTON`: quasi-Newton (BFGS) inner-weighting scheme. For each
  latent variable, jointly fits inner weights over all neighbors
  (predecessors and successors together) via BFGS minimization of a
  least-squares objective, in contrast to the classical PATH scheme,
  which mixes OLS coefficients for predecessors with bare correlations
  for successors. Initialized from the analytical OLS solution; uses
  scipy.optimize for the second-order Hessian-secant update.
- `plspm.fimix.FIMIX`: Finite Mixture PLS (Hahn et al. 2002) for latent
  class segmentation. EM algorithm with multiple random restarts detects
  K subgroups sharing the measurement model but with distinct structural
  paths. Reports per-class path coefficients, posterior memberships,
  hard assignments, and information criteria (AIC, AIC3, AIC4, BIC,
  CAIC, MDL5, normalized entropy EN). Exposed as `Plspm.fimix(n_classes)`.
- `plspm.ipma.IPMA`: Importance-Performance Map Analysis. For a chosen
  target endogenous LV, returns each predecessor's importance (total
  effect) and performance (mean of 0-100-rescaled LV score), plus an
  indicator-level breakdown with rescaled-mean performance and
  normalized weights. Exposed as `Plspm.ipma(target)`.
- `plspm.moderation.Moderation`: two-stage moderation
  (Henseler & Chin 2010). Fits a base model, multiplies the standardized
  LV scores for predictor and moderator into a product column, and
  refits with that product as a single-indicator construct pointing at
  the target. Exposes `base()`, `refit()`, and `interaction_effect()`.
- `plspm.predict.PLSPredict`: PLSpredict via k-fold cross-validation.
  Per-indicator RMSE/MAE for PLS and a linear-regression benchmark,
  plus Q²_predict against the indicator-average baseline (Shmueli et
  al. 2019). Exposed as `Plspm.predict(k=10, repeats=1, seed=42)`;
  `summary()` returns the per-indicator PLS-vs-LM verdict.

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

[Unreleased]: https://github.com/jojacobsen/openpls-engine/compare/v0.7.0a3...HEAD
[0.7.0a3]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a3
[0.7.0a2]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a2
[0.7.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a1
[0.6.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.6.0a1
