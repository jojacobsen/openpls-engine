<p align="center">
  <img src="assets/logo.svg" alt="openpls engine" width="320">
</p>

<p align="center">
  <strong>A modern, maintained Python engine for Partial Least Squares Structural Equation Modeling (PLS-SEM).</strong>
</p>

<p align="center">
  <a href="https://github.com/jojacobsen/openpls-engine/actions/workflows/ci.yml"><img src="https://github.com/jojacobsen/openpls-engine/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/openpls-engine/"><img src="https://img.shields.io/pypi/v/openpls-engine.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/openpls-engine/"><img src="https://img.shields.io/pypi/pyversions/openpls-engine.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPLv3+-blue.svg" alt="License: GPL-3.0-or-later"></a>
  <a href="https://doi.org/10.5281/zenodo.20509385"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20509385.svg" alt="DOI"></a>
  <a href="https://doi.org/10.5281/zenodo.20511533"><img src="https://img.shields.io/badge/validation%20data-10.5281%2Fzenodo.20511533-blue" alt="Validation DOI"></a>
</p>

<p align="center">
  <a href="https://openpls.app/engine">Documentation</a>
  &nbsp;·&nbsp;
  <a href="https://openpls.app/engine/quickstart/">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="https://openpls.app/engine/api/">API reference</a>
  &nbsp;·&nbsp;
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

`openpls-engine` is the compute core behind [OpenPLS](https://openpls.app) and a standalone PyPI package. It is a maintained fork of [`plspm-python`](https://github.com/googlecloudplatform/plspm-python) by Jez Humble (Google), with the original algorithm kept intact and modern PLS-SEM reporting, advanced analyses, and two new inner-weighting schemes layered on top.

> **Status: stable as of 1.0.0.** Public API follows semver. See [CHANGELOG.md](CHANGELOG.md) for the version history.

## Install

```sh
pip install openpls-engine
```

Pin a specific version for reproducible analyses:

```sh
pip install openpls-engine==1.4.0
```

Or work from source:

```sh
git clone https://github.com/jojacobsen/openpls-engine.git
cd openpls-engine
python3 -m pip install -e .
```

## Quickstart

```py
import pandas as pd
from openpls import Plspm
import openpls.config as c
from openpls.scheme import Scheme
from openpls.mode import Mode

satisfaction = pd.read_csv("tests/data/satisfaction.csv", index_col=0)

structure = c.Structure()
structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
structure.add_path(["QUAL"], ["VAL", "SAT"])
structure.add_path(["VAL"],  ["SAT"])
structure.add_path(["SAT"],  ["LOY"])

config = c.Config(structure.path(), scaled=False)
for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
    config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())

fit = Plspm(satisfaction, config, Scheme.CENTROID)
print(fit.inner_summary())
print(fit.path_coefficients())
print(fit.report().reliability())  # alpha, rho_A, rho_C, AVE
```

See the [Quickstart guide](https://openpls.app/engine/quickstart/) for the full walkthrough.

## What's inside

**Quality criteria** — HTMT and HTMT2 (geometric-mean refinement, Roemer et al. 2021), SRMR, d_ULS, Cronbach α, Dijkstra-Henseler ρ_A and ρ_C, adjusted R², BIC, Stone-Geisser Q² (blindfolding), Cohen f² effect sizes, Fornell-Larcker discriminant validity, per-indicator and per-predictor VIF, CTA-PLS (confirmatory tetrad analysis for reflective measurement).

**Advanced analyses** — PLSc (consistent-PLS bias correction, Dijkstra & Henseler 2015), Gaussian-copula endogeneity test (Park & Gupta 2012; Hult et al. 2018), disjoint two-stage higher-order constructs covering all four canonical types (R-R / R-F / F-R / F-F), IPMA (Importance-Performance Map Analysis), PLSpredict with the complete Shmueli et al. 2019 panel (RMSE / MAE / MAPE, in-sample + out-of-sample, PLS + LM benchmark), two-stage moderation, FIMIX-PLS finite-mixture segmentation, specific indirect effects with bootstrap percentile CIs, multi-group analysis with Henseler permutation tests.

**Engine internals** — five inner-weighting schemes (Centroid, Factorial, Path plus the new **quasi-Newton/BFGS** and **Lohmöller PCA** schemes), mean-replacement missing-value strategy alongside the upstream casewise default, long-running bootstrap with BCa percentile CIs and progress streaming, publication-ready `Plspm.report()` that bundles every reviewer-standard panel for one-call export.

## Documentation

Full docs at [**openpls.app/engine**](https://openpls.app/engine):

- [Introduction](https://openpls.app/engine/) — what the library does and who it is for.
- [Installation](https://openpls.app/engine/installation/) — pip and source installs.
- [Quickstart](https://openpls.app/engine/quickstart/) — end-to-end fit on the satisfaction dataset.
- [Core concepts](https://openpls.app/engine/concepts/) — PLS-SEM, Mode A/B, inner-weighting schemes, missing-value strategies.
- [API reference](https://openpls.app/engine/api/) — the full public surface.
- [Examples](https://openpls.app/engine/examples/) — runnable snippets for each advanced analysis.
- [Changelog](https://openpls.app/engine/changelog/) — version history with per-feature notes.

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

pytest          # run the test suite
ruff check .    # lint
```

CI runs lint and tests on Python 3.10 through 3.13 against every push and pull request against `main`.

## Versioning

`openpls-engine` follows [Semantic Versioning](https://semver.org). The public API is stable as of `1.0.0`. Tagged releases (`vX.Y.Z`) trigger a GitHub Actions workflow that builds the package and publishes it to PyPI via OIDC trusted publishing. The version is the single source of truth in [`pyproject.toml`](pyproject.toml) and is exposed at runtime as `openpls.__version__`. `1.0.0` renamed the import namespace from `plspm` to `openpls`; consumers upgrading from `0.7.x` must rewrite their imports (see CHANGELOG → Breaking).

## Cite

If you use `openpls-engine` in academic work, please cite the software via its [Zenodo DOI](https://doi.org/10.5281/zenodo.20509385) and the validation dataset via [10.5281/zenodo.20511533](https://doi.org/10.5281/zenodo.20511533). See [`CITATION.cff`](CITATION.cff) for the structured metadata.

## License

GNU General Public License v3.0, see [LICENSE](LICENSE). Inherited from upstream `plspm-python` (also GPL-3.0).

## Attribution

This project is a fork of [googlecloudplatform/plspm-python](https://github.com/googlecloudplatform/plspm-python) by Jez Humble. The upstream R package [plspm](https://github.com/gastonstat/plspm) by Gaston Sanchez and the [seminr](https://github.com/sem-in-r/seminr) package by Soumya Ray and Nicholas Danks remain the conceptual references for the algorithm. See [ATTRIBUTION.md](ATTRIBUTION.md) for details.

OpenPLS is an independent project and not affiliated with Google.
