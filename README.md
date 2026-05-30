# OpenPLS Engine

The compute engine behind [OpenPLS](https://openpls.app) — a Python 3 library
for Partial Least Squares Structural Equation Modeling (PLS-SEM).

`openpls-engine` is a fork of the
[`plspm-python`](https://github.com/googlecloudplatform/plspm-python) package
by Jez Humble (Google). It keeps the original algorithm intact and adds the
metrics and quality criteria that modern PLS-SEM reporting requires: HTMT,
SRMR, d_ULS, adjusted R², BIC, Q², and more.

The engine is also the core that ships in [OpenPLS](https://openpls.app) — the
hosted web application — and powers the CLI / Docker self-host distribution
planned for Phase 5 of the OpenPLS roadmap.

> **Status: pre-release / baseline.** This repo currently mirrors
> `plspm-python` 0.5.7 with attribution. The OpenPLS-specific extensions
> (metrics, MGA, bootstrap helpers) live in the OpenPLS web-app repository and
> are being migrated here step by step. See [TODO.md](TODO.md) for the roadmap.

## Why fork

* Upstream has not seen a release since 2020 (last commit June 2024).
* OpenPLS adds substantial extensions: SRMR, d_ULS, HTMT, Q², adjusted R²,
  BIC, multi-group analysis (MGA), permutation tests, and more.
* The OpenPLS hosted product depends on a single source of truth for the
  algorithm; a maintained, versioned package makes that practical.
* The intention is to release `openpls-engine` on PyPI under the
  `openpls-engine` name once the API stabilizes (see TODO).

## Installation

The package is not yet on PyPI under the new name. To use it from source:

```sh
git clone https://github.com/jojacobsen/openpls-engine.git
cd openpls-engine
python3 -m pip install -e .
```

## Usage

The public API mirrors upstream `plspm` 0.5.7. The
[upstream documentation](https://plspm.readthedocs.io/) is still authoritative
for the current code; OpenPLS-specific docs will land alongside the metric
extensions.

```py
import pandas as pd
import plspm.config as c
from plspm.plspm import Plspm
from plspm.scheme import Scheme
from plspm.mode import Mode

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

GNU General Public License v3.0 — see [LICENSE](LICENSE). Inherited from
upstream `plspm-python` (also GPL-3.0).

## Attribution

This project is a fork of
[googlecloudplatform/plspm-python](https://github.com/googlecloudplatform/plspm-python)
by Jez Humble. The upstream R package
[plspm](https://github.com/gastonstat/plspm) by Gaston Sanchez and the
[seminr](https://github.com/sem-in-r/seminr) package by Soumya Ray and Nicholas
Danks remain the conceptual references for the algorithm.

OpenPLS is an independent project and not affiliated with Google.
