# OpenPLS Engine — Roadmap

The engine is in **baseline** state: it is currently a 1:1 mirror of
`plspm-python` 0.5.7 with the OpenPLS rebrand applied to README and metadata.
The work below moves it toward a maintained, full-featured PLS-SEM engine that
can replace `pip install plspm` in the OpenPLS web app and serve as a
standalone PyPI package.

The items below are roughly ordered by how OpenPLS itself depends on them.
Each item should ship as a separate PR with tests.

## Phase A — Migrate OpenPLS extensions back upstream

The OpenPLS web-app repo has already accumulated several extensions on top of
`plspm`. They live in `functions/compute/` of the web-app repo and need to be
ported here so the engine becomes self-contained.

- [x] **SRMR + d_ULS** (Standardized Root Mean Square Residual,
      unweighted least-squares discrepancy). Ported in `plspm.fit.ModelFit`
      (commit `d0c762c`).
- [x] **HTMT** (Heterotrait-Monotrait ratio of correlations). Ported in
      `plspm.htmt.HTMT` (commit `e5ab424`). SmartPLS cross-validation
      pending reference values from Johannes.
- [x] **Adjusted R²** (already exposed by `InnerModel.r_squared_adj`) and
      **BIC** ported in `plspm.inner_summary` (commit `<this commit>`).
      Adj R² uses `1 - (1-R²)(n-1)/(n-k-1)`; BIC = `n·log(SSE/n) + (k+1)·log(n)`
      with `SSE = (1-R²)(n-1)` for standardized LV scores.
- [ ] **Q²** (blindfolding) — pending.
- [ ] **Cronbach α + Dijkstra-Henseler ρ** with the endogenous-LV fallback
      from web-app fix #65.
- [ ] **Multi-Group Analysis** (Henseler permutation) — currently in
      `functions/compute/mga.py`. Should land as a top-level `plspm.mga`
      submodule.
- [ ] **Mean replacement** option for missing values (web-app fix #66).
- [ ] **Bootstrap multi-core helper** — upstream already has
      `bootstrap.py`; OpenPLS adds a long-running, resumable variant for
      Cloud-Run-style workloads.

## Phase B — Numerical alignment with SmartPLS / R `plspm`

OpenPLS has open issues around small numerical drift vs SmartPLS for path
coefficients and SRMR on some validation cases. These need root-cause analysis
and matching upstream conventions before a 1.0 release.

- [ ] **Investigate plspm-vs-SmartPLS drift** in paths and SRMR (OpenPLS issue
      #67). Likely candidates: inner-weighting scheme tie-breaking, indicator
      standardization edge cases.
- [ ] **Normalize outer weights** to the SmartPLS convention (signs and
      scaling); currently weights match `plspm` but not always SmartPLS
      (OpenPLS issue #69).

## Phase C — Algorithmic features

The features below are not in `plspm` 0.5.7 today. They are required to call
OpenPLS Engine feature-complete vs SmartPLS 4 / ADANCO.

- [ ] **PCA inner weighting scheme** (Lohmöller) — alternative to centroid,
      factorial, and path.
- [ ] **PLSpredict / Q²-Predict** — out-of-sample predictive power.
- [ ] **IPMA** (Importance-Performance Map Analysis) — common output in
      applied marketing/IS research.
- [ ] **FIMIX-PLS** (Finite Mixture Segmentation) — latent class segmentation
      for unobserved heterogeneity.
- [ ] **Moderation / interaction terms** — two-stage and product-indicator
      approaches.

## Phase D — Packaging + distribution

- [ ] Rename the Python package itself from `plspm` to `openpls_engine`
      (currently the directory is still `plspm/` to keep the OpenPLS web app
      running while we migrate). Provide a `plspm` shim module for backwards
      compatibility during the transition.
- [ ] Switch `setup.py` → `pyproject.toml` (PEP 621).
- [ ] **Publish to PyPI** as `openpls-engine` once Phase A is complete
      and the API has stabilized.
- [ ] CI on GitHub Actions: lint (ruff), type-check (mypy), tests (pytest)
      against Python 3.10–3.13.
- [ ] Versioning: SemVer. Pre-1.0 releases on `0.x` while migration runs.

## Phase E — Self-host distribution

- [ ] Standalone CLI: `openpls-engine run model.json data.csv` — emits a
      JSON / Markdown result document equivalent to the web app's results
      panel.
- [ ] Docker image for batch / self-host workloads (Phase 5 of the OpenPLS
      product roadmap).

## Out-of-scope (for now)

- A Web UI lives in the closed-source OpenPLS web-app repo. The engine stays
  a library, with no UI dependencies.
- R compatibility (a CRAN package) — not planned.
