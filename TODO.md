# OpenPLS Engine, Roadmap

The engine started as a 1:1 mirror of `plspm-python` 0.5.7 with an OpenPLS
rebrand on README and metadata. The work below moves it toward a maintained,
full-featured PLS-SEM engine that can replace `pip install plspm` in the
OpenPLS web app and serve as a standalone PyPI package.

The items below are roughly ordered by how OpenPLS itself depends on them.
Each item should ship as a separate PR with tests.

## Phase A, Migrate OpenPLS extensions back upstream (complete)

The OpenPLS web-app repo had accumulated several extensions on top of
`plspm`. They lived in `functions/compute/` of the web-app repo and have now
been ported here so the engine is self-contained.

- [x] **SRMR + d_ULS** (Standardized Root Mean Square Residual,
      unweighted least-squares discrepancy). Ported in `plspm.fit.ModelFit`
      (commit `d0c762c`).
- [x] **HTMT** (Heterotrait-Monotrait ratio of correlations). Ported in
      `plspm.htmt.HTMT` (commit `e5ab424`). Cross-validation against
      reference software pending values from Johannes.
- [x] **Adjusted R²** (already exposed by `InnerModel.r_squared_adj`) and
      **BIC** ported in `plspm.inner_summary`.
      Adj R² uses `1 - (1-R²)(n-1)/(n-k-1)`; BIC = `n·log(SSE/n) + (k+1)·log(n)`
      with `SSE = (1-R²)(n-1)` for standardized LV scores.
- [x] **Q²** (Stone-Geisser, cross-validated redundancy via blindfolding).
      Ported in `plspm.q_squared.QSquared`, exposed as `Plspm.q_squared()`.
      ECSI baseline at D=7: EXPE 0.20, QUAL 0.47, VAL 0.38, SAT 0.51, LOY 0.33.
- [x] **Cronbach α + Dijkstra-Henseler ρ** with listwise-deletion fallback
      from web-app fix #65. Ported in `plspm.unidimensionality.Unidimensionality`:
      when an LV's indicator block contains NaN, rows are dropped pairwise and a
      per-block correction is applied. Upstream returned NaN for the whole block.
- [x] **Multi-Group Analysis** (Henseler permutation) ported as `plspm.mga`
      submodule with classes `MGA` and `GroupSpec`. Supports categorical
      (`values=[...]`) and numeric range (`range=(lo, hi)`) group definitions,
      2+ groups with all pairwise comparisons, two-sided permutation p-values
      with Phipson-Smyth (2010) add-one smoothing.
- [x] **Mean replacement** option for missing values (web-app fix #66).
      Ported as `Plspm(..., missing_strategy="mean")`. Default `"casewise"`
      preserves upstream behavior. Mean strategy fills NaN in every indicator
      column with the column mean before estimation, matching the
      "Mean replacement" option in commercial PLS-SEM software.
- [x] **Long-running bootstrap helper**. Upstream `bootstrap.py` favours
      multiprocessing for short runs. OpenPLS adds
      `plspm.long_bootstrap.LongBootstrap`, a serial variant with progress
      callback, sign-flipping, BCa percentile CIs, normal-approximation
      p-values, and a configurable success-rate floor. Suited for
      Cloud-Run-style workloads that stream progress to Firestore.

## Phase B, Numerical alignment with reference implementations

OpenPLS has open issues around small numerical drift versus established
PLS-SEM software for path coefficients and SRMR on some validation cases.
These need root-cause analysis and matching established conventions before a
1.0 release.

- [ ] **Investigate path/SRMR drift versus reference implementations**
      (OpenPLS issue #67). Likely candidates: inner-weighting scheme
      tie-breaking, indicator standardization edge cases.
- [ ] **Normalize outer weights** to the convention used by mainstream
      PLS-SEM tools (signs and scaling); currently weights match `plspm` but
      not always the established convention (OpenPLS issue #69).

## Phase C, Algorithmic features

The features below are not in `plspm` 0.5.7 today. They are required to call
OpenPLS Engine feature-complete with mainstream PLS-SEM software.

- [x] **Quasi-Newton inner weighting scheme**, an alternative to
      centroid, factorial, and path. Exposed as `Scheme.NEWTON`: for each
      LV, jointly fits inner weights over all neighbors (predecessors and
      successors together) via BFGS optimization of a least-squares
      objective, in contrast to PATH which mixes OLS coefficients for
      predecessors with correlations for successors. The originally
      planned Lohmöller PCA scheme was deprioritized in favour of this
      genuinely second-order alternative.
- [x] **PLSpredict / Q²-Predict**, out-of-sample predictive power.
      Ported as `plspm.predict.PLSPredict`, exposed as `Plspm.predict()`.
      k-fold cross-validation; per-indicator RMSE/MAE for PLS and a linear
      regression benchmark, plus Q²_predict against the indicator-average
      baseline. `summary()` classifies each indicator as PLS-better/worse/tie.
- [x] **IPMA** (Importance-Performance Map Analysis), a common output in
      applied marketing and IS research. Ported as `plspm.ipma.IPMA`,
      exposed as `Plspm.ipma(target)`. Returns LV-level and indicator-level
      importance/performance tables; performance rescaled to 0-100 from
      observed or supplied indicator scale.
- [x] **FIMIX-PLS** (Finite Mixture Segmentation), latent class
      segmentation for unobserved heterogeneity. Ported as
      `plspm.fimix.FIMIX`, exposed as `Plspm.fimix(n_classes)`. EM with
      multiple random restarts; reports per-class path coefficients,
      posterior memberships, hard assignments, and model-selection
      criteria (AIC, AIC3, AIC4, BIC, CAIC, MDL5, normalized entropy EN).
- [x] **Moderation / interaction terms** via the two-stage approach
      (Henseler & Chin 2010). Ported as `plspm.moderation.Moderation`:
      fits a base model, takes standardized LV scores for predictor and
      moderator, multiplies them, and refits with the product as a
      single-indicator construct pointing at the target.

## Phase D, Packaging + distribution

- [x] PEP 621 `pyproject.toml`.
- [x] CI on GitHub Actions: lint (ruff), tests (pytest) against
      Python 3.10 through 3.13.
- [x] SemVer versioning. Pre-1.0 releases on `0.x` while migration runs.
- [x] Release workflow: tag `v*` triggers PyPI publish via OIDC trusted
      publishing.
- [ ] Rename the Python package itself from `plspm` to `openpls_engine`
      (currently the directory is still `plspm/` to keep the OpenPLS web app
      running while we migrate). Provide a `plspm` shim module for
      backwards compatibility during the transition.
- [ ] Add mypy type-check to CI.
- [ ] **Publish to PyPI** as `openpls-engine` once Phase B is stable and the
      API has settled.

## Phase E, Self-host distribution

- [ ] Standalone CLI: `openpls-engine run model.json data.csv`, emits a
      JSON / Markdown result document equivalent to the web app's results
      panel.
- [ ] Docker image for batch / self-host workloads (Phase 5 of the OpenPLS
      product roadmap).

## Out-of-scope (for now)

- A Web UI lives in the closed-source OpenPLS web-app repo. The engine stays
  a library, with no UI dependencies.
- R compatibility (a CRAN package), not planned.
