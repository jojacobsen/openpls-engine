# Changelog

All notable changes to `openpls-engine` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0 releases live on the `0.x` line while the API stabilizes. Tagged
releases (`vX.Y.Z`) trigger a GitHub Actions workflow that builds the
package and publishes it to PyPI via OIDC trusted publishing.

## [1.9.0] - 2026-06-19

Addresses two engine-side items from Florian Schuberth's technical review
of OpenPLS (2026-06-18): HTMT/HTMT2 are now correctly gated to reflective
constructs, and the `Unidimensionality` class has been renamed to
`Reliability` to match what it actually measures.

### Changed
- **HTMT and HTMT2** now return `NaN` for any latent-variable pair that
  involves a formative (Mode B) construct. Both metrics estimate the
  correlation between *reflectively measured* latent variables; applying
  them to formative blocks produced misleading discriminant-validity
  numbers. The reflective-only restriction is now enforced inside
  `openpls.htmt.HTMT` and `openpls.htmt2.HTMT2`. Mode A pairs are
  unaffected.
- **`Unidimensionality` → `Reliability`** (#schuberth-review). The class
  that computes Cronbach's α and Dillon-Goldstein ρ_c has been renamed to
  `openpls.reliability.Reliability`; those metrics measure reliability,
  not unidimensionality (which is a stricter psychometric concept).
  `Plspm.reliability()` is the new accessor.

### Deprecated
- **`openpls.unidimensionality.Unidimensionality`** and
  **`Plspm.unidimensionality()`** now emit `DeprecationWarning` and forward
  to the new names. Behaviour is identical; the old names will be removed
  in a future major release.

## [1.8.0] - 2026-06-13

Maintenance release. No engine API changes, no behavioral changes. Fixes the
PyPI project page (its Markdown renderer cannot resolve repo-relative links)
and locks the developer / CI / Read-the-Docs environment to a known-good set
of dependency versions verified to install across Python 3.10–3.13.

### Changed
- **PyPI project page** now links the Changelog to
  `https://openpls.app/engine/changelog/`. PyPI strips repo-relative
  `CHANGELOG.md` links from rendered READMEs, so the previous link silently
  vanished on the project page. The `[project.urls]` table in
  `pyproject.toml` also gains an explicit `Changelog` entry which surfaces
  in the PyPI sidebar.
- **`requirements.txt`** now pins exact versions for the developer
  toolchain (pandas, numpy, scipy, statsmodels, scikit-learn, pytest,
  setuptools, wheel, twine, sphinx, m2r) instead of unpinned floats. The
  install metadata in `pyproject.toml` keeps the loose floor ranges so
  consumers are not over-constrained. The pinned set is the matrix the CI
  job and the Read-the-Docs build run against on every push.

## [1.7.0] - 2026-06-12

Extends the PLSc consistent-PLS layer with first-class direct / indirect /
total effects and specific indirect effects, so mediation analysis on a
common-factor model no longer carries the composite-model attenuation
forward into the chain product.

### Added
- **`PLSc.effects()`** returns a direct / indirect / total effects table
  computed by walking the structural DAG over the dis-attenuated path
  matrix. Drop-in replacement for `Plspm.effects()` when interpreting the
  model as a common-factor (covariance-based) one — same column layout
  (`from`, `to`, `direct`, `indirect`, `total`) and index labels
  (`"A -> B"`).
- **`PLSc.specific_indirect_effects(source, target, through=None)`**
  point-estimates each `source -> ... -> target` mediation chain by
  multiplying the PLSc β along the chain (Zhao/Lynch/Chen 2010;
  Nitzl/Roldán/Cepeda 2016). Without disattenuation the SIE inherits the
  measurement-error attenuation that PLSc was designed to remove. Same
  interface as `Plspm.specific_indirect_effects` (chain enumeration,
  explicit `through=` path, error cases).

## [1.6.0] - 2026-06-12

Adds bootstrap-based multi-group analysis for the two-group comparison case
(complementing the permutation-based `MGA`), the canonical disattenuated
PLSc quality panel (AVE / ρ_c / SRMR / BIC / HTMT / VIF on the corrected
metric), Henseler-convention IPMA normalized weights, MICOM's raw
variance-difference statistic, PLSpredict's earliest-antecedents LM
benchmark, and unified bootstrap inference tables on `LongBootstrap`.

### Added
- **Bootstrap-based Multi-Group Analysis** via
  `openpls.bootstrap_mga.BootstrapMGA`. Complements the permutation-based
  `openpls.mga.MGA` with bootstrap inference on the difference between
  exactly two groups. For every path coefficient, outer loading, outer
  weight, total effect, and specific / total indirect effect, three pairwise
  difference tests are reported side by side:
  - **Henseler (2007) distribution-based** one- / two-tailed p-values
    derived from the rank position of one group's bootstrap mean in the
    other group's resample distribution.
  - **Chin (2000) parametric** pooled-variance t-test on bootstrap
    standard errors (Hair Primer 3rd ed. Eq. 4.7).
  - **Welch-Satterthwaite** unequal-variance variant (Eq. 4.8) for groups
    with heteroscedastic resample distributions.
  Each panel also returns per-group bootstrap means, standard errors, and
  BCa percentile confidence intervals (Efron 1987). Resamples follow the
  PLS sign indeterminacy and remain unflipped to match SmartPLS's "no sign
  changes" option.
- **Unified bootstrap inference tables** on `LongBootstrap` via the new
  `inference` property. Returns a dict keyed by `pathCoefficients`,
  `outerLoadings`, `outerWeights`, `specificIndirectEffects`,
  `totalIndirectEffects`, and `totalEffects`. Each DataFrame carries the
  canonical inference columns `original`, `mean`, `std_error`, `t_value`,
  `p_value`, `ci_percentile_2_5`, `ci_percentile_97_5`, `ci_bc_2_5`,
  `ci_bc_97_5`. The bias-corrected CI uses the Efron (1987) z₀ formula and
  the p-value is computed from the two-sided recentred bootstrap
  distribution (Davison & Hinkley 1997 §4.4). Resamples remain accessible
  via the new `resamples`, `path_keys`, `outer_keys`, and `lv_names`
  properties for downstream analyses (MGA, custom tests).

### Changed
- `LongBootstrap.paths()/loadings()/weights()/total_effects()` now expose
  both percentile and bias-corrected CI columns alongside the existing
  `ci_lower`/`ci_upper` (which continue to hold the BC bounds for
  backwards compatibility).
- `IPMA.indicators()` now emits two additional columns to match the
  Henseler IPMA convention (Ringle & Sarstedt 2016):
  `indicator_importance` (= `outer_weight × lv_importance`, the
  indicator's total effect on the IPMA target) and
  `henseler_normalized_weight` (= `indicator_importance / lv_importance`,
  the value SmartPLS-style IPMA tables report under "Normalized Weight").
  The legacy `normalized_weight` (weight ÷ Σ weight per LV) remains for
  backwards compatibility.
- `MICOM.step3()` now emits an additional `var_diff` column (= `var_a -
  var_b`) alongside the existing `log_var_ratio` (= `log(var_a / var_b)`,
  Henseler/Ringle/Sarstedt 2016 §3.4). Both quantities are zero under H₀
  and share the same sign; the raw difference matches the convention used
  by SmartPLS-style validation tables, while the log-ratio remains the
  canonical Henseler test statistic.
- `Plspm.predict()` / `PLSPredict` accepts a new `lm_predictor_set`
  parameter selecting the LM benchmark's regressor block:
  - `"direct"` (default, backwards-compatible): the LV's direct path
    predecessors.
  - `"earliest_antecedents"`: walks upstream through every mediator and
    uses only the exogenous LVs at the top of the structural DAG, per the
    Shmueli/Hair/Ringle 2019 PLSpredict convention. The PLS-side
    predictions are unaffected; only the LM benchmark moves.
- `PLSc` now exposes the full disattenuated reflective-LV quality panel,
  closing the gap that previously mixed PLSc paths/loadings with
  PLS-SEM AVE/ρ_c/SRMR/BIC/HTMT/VIF (Dijkstra & Henseler 2015):
  - `PLSc.ave()` — Average Variance Extracted on the PLSc loadings,
    `λ_c = w · sqrt(rho_A) / (w'w)`.
  - `PLSc.rho_c()` — Jöreskog's composite reliability evaluated on the
    PLSc loadings.
  - `PLSc.htmt()` — disattenuated construct-correlation magnitude
    (consistent estimator of the latent correlation under congeneric
    reflective measurement; same 0.85 / 0.90 thresholds as HTMT).
  - `PLSc.srmr()` / `PLSc.d_uls()` — saturated-model fit on the PLSc
    implied indicator correlation matrix `Σ̂_c = Λ_c Φ_c Λ_cᵀ`, where
    `Φ_c` is the dis-attenuated LV correlation matrix.
  - `PLSc.bic()` — Bayesian Information Criterion per endogenous LV
    using the corrected R².
  - `PLSc.vif_inner()` — inner VIF per endogenous LV computed on the
    dis-attenuated correlation metric.
  - `PLSc.summary()` now additionally returns `ave`, `rho_c`, and `bic`
    columns (existing `rho_a`, `r_squared`, `r_squared_adj` unchanged).
  The PLS-SEM `Plspm` accessors (`inner_summary`, `model_fit`, `vif`,
  `htmt`, …) still return the uncorrected composite-model values so
  both interpretations remain available side-by-side.

## [1.5.0] - 2026-06-10

Adds the canonical pre-MGA measurement-invariance check so engine users can
verify that composite constructs are comparable across two groups before
interpreting group differences. Single additive feature, no existing
behaviour changes.

### Added
- **MICOM — Measurement Invariance of Composite Models**
  (Henseler, Ringle & Sarstedt 2016) via `Plspm.micom()` →
  `openpls.micom.MICOM`. Runs the three-step procedure:
  - **Step 1 — Configural invariance:** guaranteed by reusing one
    `Config` across both groups (audit trail via `MICOM.group_sizes()`).
  - **Step 2 — Compositional invariance:** per construct, the correlation
    `c = w_A' Σ w_B / sqrt((w_A' Σ w_A)(w_B' Σ w_B))` between group-A and
    group-B weights evaluated on the pooled indicator covariance, tested
    against `c = 1` with a one-sided lower-tail permutation test. Sign
    indeterminacy is handled by aligning each permutation's weight
    direction before computing `c`.
  - **Step 3 — Equality of composite means and variances:** pooled-fit
    weights applied to standardized indicators produce common-scale
    composite scores; mean differences and `log(var_A / var_B)` are then
    tested with two-sided label-shuffling permutations (no PLS refit per
    iteration, so this step is cheap).
  - `MICOM.summary()` collapses the three steps into a per-construct
    verdict: `"full"` (Step 2 + Step 3 pass), `"partial"` (Step 2 passes
    but mean or variance differs), or `"none"` (Step 2 fails — composites
    are not comparable and MGA results would be uninterpretable).
  Closes a longstanding gap: prior releases supported MGA but provided no
  in-engine way to verify the invariance prerequisite MGA assumes.

## [1.4.0] - 2026-06-09

Four seminr-aligned additions covering structural effect sizes,
discriminant validity, mediation decomposition, predictive accuracy, and
one-call reporting. All APIs are additive (no existing behaviour or
signatures changed), so this is a minor bump.

### Added
- **Publication-ready summary report** via `Plspm.report()` →
  `openpls.report.Report`. Bundles the engine's individual diagnostics
  (reliability with Cronbach α / ρ_A / ρ_C / AVE, HTMT and HTMT2 matrices
  plus long-form pair tables, Fornell-Larcker matrix and per-LV verdict,
  structural paths with f² and effect-size labels, per-LV R² / adjusted
  R² / BIC / block communality / mean redundancy, SRMR / d_ULS / GoF,
  outer and inner VIF) into one object so the standard PLS-SEM research
  report (Hair, Hult, Ringle & Sarstedt 2022) can be exported with a
  single call. Pure orchestration over existing `Plspm` methods — no new
  numbers, no new behavior.
- **Specific indirect effects** via `Plspm.specific_indirect_effects()`.
  Per-path mediation decomposition along every directed predecessor →
  successor route in the structural model, with bootstrap percentile CIs
  and significance verdicts. Aligns the engine with `seminr` 's
  `specific_effect_significance()` (Hair et al. 2022 §7.3).
- **PLSpredict full Shmueli et al. 2019 panel.** `PLSPredict.summary()`
  now reports the complete reviewer-standard table: in-sample +
  out-of-sample RMSE, MAE, and MAPE for both PLS and the LM benchmark,
  plus the PLS-vs-LM verdict per indicator. The existing one-sided
  Q²_predict against the indicator-average baseline is unchanged.
- **Cohen f² effect sizes** via `Plspm.f_squared()` →
  `openpls.f_squared.FSquared`. Per-path effect-size decomposition with
  the canonical small / medium / large thresholds (0.02 / 0.15 / 0.35,
  Cohen 1988; Hair et al. 2022). Aligns with `seminr::f2()`.
- **Fornell-Larcker discriminant validity** via `Plspm.fornell_larcker()`
  → `openpls.fornell_larcker.FornellLarcker`. Matrix view with √AVE on
  the diagonal and inter-construct correlations off-diagonal, plus a
  per-LV verdict that flags any construct whose √AVE is below its
  largest inter-construct correlation. Aligns with `seminr` 's
  `fornell_larcker()` reporting.

## [1.3.0] - 2026-06-09

Disjoint two-stage higher-order constructs as a first-class API. The
existing legacy `Config.add_higher_order` stays untouched, so this is a
minor bump.

### Added
- **Disjoint two-stage higher-order constructs (HOC)** via
  `Plspm.higher_order()` → `openpls.higher_order.HigherOrder`. Implements
  the disjoint two-stage workflow recommended by Sarstedt, Hair, Cheah,
  Becker & Ringle (2019) and Hair et al. (2022, *A Primer on PLS-SEM*,
  3rd ed., Ch. 8). The fitted `Plspm` becomes stage 1; its first-order
  LV scores are appended as indicators of the new second-order construct
  and a stage-2 `Plspm` is fit with the HOC in place of its first-order
  constituents in the structural model. All four canonical HOC types
  (Type I R-R, II R-F, III F-R, IV F-F) are covered by combining the
  first-order LV modes with the HOC mode. The legacy
  `Config.add_higher_order` (repeated-indicators / embedded two-stage)
  remains for backward compatibility but is no longer the recommended
  path.

## [1.2.0] - 2026-06-09

Three seminr-aligned diagnostics for measurement-error correction,
discriminant validity, and structural-equation endogeneity. All APIs
are additive (no existing behaviour or signatures changed), so this is
a minor bump.

### Added
- **Gaussian-copula endogeneity test** via `Plspm.copula()` →
  `openpls.copula.GaussianCopula`. Park & Gupta (2012) / Hult, Hair,
  Proksch, Sarstedt, Pinkwart & Ringle (2018) procedure for detecting
  endogeneity in PLS-SEM structural equations. Augments each suspected
  predecessor LV with a copula term `P = Phi^{-1}(F_n(X))`, refits the
  endogenous LV's regression by OLS, and tests each copula coefficient
  via a non-parametric row bootstrap (same SE / t / p convention as
  `LongBootstrap`). Each suspected predictor is screened with a
  Cramér-von Mises normality test for admissibility; the `summary()`
  marks normal predictors as `copula not admissible (normal)` because
  the test cannot tell endogeneity from a Gaussian regressor.
- **HTMT2** via `Plspm.htmt2()` → `openpls.htmt2.HTMT2`. Geometric-mean
  refinement of the Heterotrait-Monotrait Ratio of Correlations
  (Roemer, Schuberth & Henseler 2021). Replaces both arithmetic means
  in the original Henseler/Ringle/Sarstedt 2015 HTMT with geometric
  means, removing the bias HTMT shows when indicator loadings within a
  block are unequal. Same API surface as `HTMT` (`matrix()`, `pairs()`).
  Pairs involving a single-indicator construct or any zero indicator
  correlation are returned as `NaN` (the geometric mean is undefined).
  Aligns with `seminr` 's HTMT2 reporting.
- **Consistent PLS (PLSc)** via `Plspm.plsc()` → `openpls.plsc.PLSc`.
  Implements the Dijkstra & Henseler (2015) bias correction for
  reflective (Mode A) constructs: per-LV `rho_A` reliability, an
  adjusted construct-correlation matrix, corrected path coefficients
  re-estimated by OLS on the dis-attenuated correlations, corrected
  `R²` / adjusted `R²`, and corrected outer loadings under a common-
  factor interpretation. Formative (Mode B) and single-indicator
  constructs receive `rho_A = 1` by convention. Aligns with
  `seminr::PLSc()`.

## [1.1.0] - 2026-06-09

Two seminr-aligned outer-model diagnostics. Both APIs are additive (no
existing behaviour or signatures changed), so this is a minor bump.

### Added
- **Variance Inflation Factor (VIF) diagnostics** via `Plspm.vif()`.
  Two views: `items()` returns per-indicator VIF within each construct
  block (collinearity diagnostic primarily for Mode B / formative
  blocks); `inner()` returns per-predictor VIF for each endogenous LV
  (structural collinearity among antecedents). Single-indicator blocks
  and single-predictor endogenous LVs are omitted (VIF undefined or
  trivially 1). Aligns the engine with `seminr::vif_items()`.
- **Confirmatory Tetrad Analysis (CTA-PLS)** via `Plspm.cta()` →
  `openpls.cta.CTAPLS`. Diagnostic for the outer model that tests
  reflective (Mode A) specification per block of four or more indicators
  using Bollen and Ting's (1993) vanishing-tetrad theorem (procedure of
  Gudergan, Ringle, Wende & Will 2008). Bootstrap-based two-sided
  p-values under H0: tau = 0 with a within-block Holm step-down
  correction. `tetrads()` returns the per-tetrad table; `summary()`
  returns the per-block verdict (`"reflective supported"` vs
  `"reflective rejected"`). Aligns with `seminr` 's CTA tooling.

## [1.0.2] - 2026-06-01

Test-suite hardening release. No API changes, no runtime behavior changes.

### Added
- **Synthetic regression tests** that do not depend on any external
  reference output:
  - `tests/test_scheme_equivalence_two_lv.py`: locks down the invariant
    that PATH, CENTROID, FACTORIAL, PCA, and NEWTON inner-weighting
    schemes produce identical path coefficients, weights, loadings, and
    R² on any two-LV model (each LV has exactly one neighbour, so the
    inner update degenerates).
  - `tests/test_redundancy_analysis_mode_b.py`: a Mode B formative
    driver block predicting a single-item global rating LV. Asserts
    path recovery is positive and within a sampling band across seeds,
    R² lies in the expected attenuated range, and the single-indicator
    loading is exactly 1.0. Also parametrized across inner schemes for
    the degenerate two-LV case.
  - `tests/test_path_recovery_synthetic.py`: three-LV mediation chain
    X → M → Y with known structural coefficients. Asserts the engine
    recovers direct paths, indirect / direct / total effects, and the
    population R² on Y within sampling tolerance over multiple seeds.

### Changed
- Docstring and comment phrasing neutralised in `openpls/config.py`,
  `openpls/fit.py`, `tests/test_fit.py`, and `tests/test_sign_convention.py`
  to reference the underlying convention (Henseler et al. 2014 §5.3,
  Hair et al., Wold) rather than naming a comparison tool. No code
  behaviour change.

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

[Unreleased]: https://github.com/jojacobsen/openpls-engine/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/jojacobsen/openpls-engine/releases/tag/v1.4.0
[0.7.0a3]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a3
[0.7.0a2]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a2
[0.7.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.7.0a1
[0.6.0a1]: https://github.com/jojacobsen/openpls-engine/releases/tag/v0.6.0a1
