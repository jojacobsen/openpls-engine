# OpenPLS Engine

[![CI](https://github.com/jojacobsen/openpls-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/jojacobsen/openpls-engine/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/openpls-engine.svg)](https://pypi.org/project/openpls-engine/)
[![Python versions](https://img.shields.io/pypi/pyversions/openpls-engine.svg)](https://pypi.org/project/openpls-engine/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPLv3+-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20509385.svg)](https://doi.org/10.5281/zenodo.20509385)
[![Validation DOI](https://img.shields.io/badge/validation%20data-10.5281%2Fzenodo.20511533-blue)](https://doi.org/10.5281/zenodo.20511533)

The compute engine behind [OpenPLS](https://openpls.app), a Python 3 library
for Partial Least Squares Structural Equation Modeling (PLS-SEM).

`openpls-engine` is a fork of
[`plspm-python`](https://github.com/googlecloudplatform/plspm-python) by Jez
Humble (Google). It keeps the original algorithm intact and adds the metrics
and quality criteria that modern PLS-SEM reporting requires: HTMT, SRMR,
d_ULS, adjusted R², BIC, Stone-Geisser Q², Cronbach α, Dijkstra-Henseler ρ,
VIF (per-indicator and per-predictor), CTA-PLS (confirmatory tetrad
analysis for reflective measurement), multi-group analysis, and a
progress-streaming long bootstrap. It also ships
**advanced** analyses that go beyond what `plspm-python` covers: PLSpredict
out-of-sample validation, IPMA (Importance-Performance Map Analysis),
two-stage moderation, FIMIX-PLS finite-mixture segmentation, and a
**quasi-Newton (BFGS) inner-weighting scheme** and a **Lohmöller PCA
scheme**, two new alternatives to the classical centroid, factorial, and
path schemes (five inner-weighting schemes total).

The engine also powers [OpenPLS](https://openpls.app) (the hosted web
application) and the CLI / Docker self-host distribution planned for the next
phase of the roadmap.

> **Status: stable as of 1.0.0.** Public API (Plspm, Config, Mode, Scheme,
> IPMA, PLSpredict, Moderation, FIMIX) follows semver. Numerical alignment
> with reference implementations is tracked in [TODO.md](TODO.md). See
> [CHANGELOG.md](CHANGELOG.md) for the version history.

## Documentation

Hosted docs at [**openpls.app/engine**](https://openpls.app/engine):

- [Introduction](https://openpls.app/engine/) - what the library does and who it is for.
- [Installation](https://openpls.app/engine/installation/) - pip and source installs.
- [Quickstart](https://openpls.app/engine/quickstart/) - end-to-end fit on the satisfaction dataset.
- [Core concepts](https://openpls.app/engine/concepts/) - PLS-SEM, Mode A/B, inner-weighting schemes, missing-value strategies.
- [API reference](https://openpls.app/engine/api/) - the full public surface (Plspm, Config, VIF, CTAPLS, IPMA, PLSpredict, Moderation, FIMIX, MGA, LongBootstrap).
- [Examples](https://openpls.app/engine/examples/) - runnable snippets for each advanced analysis.
- [Changelog](https://openpls.app/engine/changelog/) - version history with per-feature notes.

## Why fork

* Upstream has not seen a release since 2020.
* OpenPLS adds substantial extensions: SRMR, d_ULS, HTMT, Q², adjusted R²,
  BIC, multi-group analysis (MGA), permutation tests, mean replacement, and a
  long-running bootstrap with BCa confidence intervals.
* The OpenPLS hosted product depends on a single source of truth for the
  algorithm; a maintained, versioned package makes that practical.

## Installation

`openpls-engine` is published on PyPI:

```sh
pip install openpls-engine
```

Or pin a specific version for reproducible analyses:

```sh
pip install openpls-engine==1.0.2
```

To work from source instead:

```sh
git clone https://github.com/jojacobsen/openpls-engine.git
cd openpls-engine
python3 -m pip install -e .
```

## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

pytest          # run the test suite
ruff check .    # lint
```

CI runs lint and tests on Python 3.10 through 3.13 against every push and
pull request against `main`.

## Versioning

`openpls-engine` follows [Semantic Versioning](https://semver.org). The
public API is stable as of `1.0.0`. Tagged releases (`vX.Y.Z`) trigger a
GitHub Actions workflow that builds the package and publishes it to PyPI
via OIDC trusted publishing. The version is the single source of truth in
[`pyproject.toml`](pyproject.toml) and is exposed at runtime as
`openpls.__version__`.

See [CHANGELOG.md](CHANGELOG.md) for the per-version history. `1.0.0`
renamed the import namespace from `plspm` to `openpls`; consumers upgrading
from `0.7.x` must rewrite their imports (see CHANGELOG → Breaking).

## Usage

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
structure.add_path(["VAL"], ["SAT"])
structure.add_path(["SAT"], ["LOY"])

config = c.Config(structure.path(), scaled=False)
for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
    config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())

result = Plspm(satisfaction, config, Scheme.CENTROID)
print(result.inner_summary())
print(result.path_coefficients())
```

## License

GNU General Public License v3.0, see [LICENSE](LICENSE). Inherited from
upstream `plspm-python` (also GPL-3.0).

## Attribution

This project is a fork of
[googlecloudplatform/plspm-python](https://github.com/googlecloudplatform/plspm-python)
by Jez Humble. The upstream R package
[plspm](https://github.com/gastonstat/plspm) by Gaston Sanchez and the
[seminr](https://github.com/sem-in-r/seminr) package by Soumya Ray and
Nicholas Danks remain the conceptual references for the algorithm. See
[ATTRIBUTION.md](ATTRIBUTION.md) for details.

OpenPLS is an independent project and not affiliated with Google.
