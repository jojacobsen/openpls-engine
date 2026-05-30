# Contributing to OpenPLS Engine

Thanks for your interest in contributing. This project is GPL-3.0-or-later,
so every contribution stays under those terms.

## Workflow

1. Open an issue describing the change before starting non-trivial work.
2. Fork the repo, create a feature branch, and open a pull request against
   `main`.
3. Each PR should ship a single self-contained change with tests. Keep
   commits focused; squash on merge if the history is messy.
4. CI runs lint (ruff) and tests (pytest) on Python 3.10 through 3.13. The
   build must be green before review.

## Local setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

pytest          # run the test suite
ruff check .    # lint
```

## Coding conventions

- Match the style of surrounding code; ruff config in `pyproject.toml` is
  authoritative.
- New numeric or statistical features ship with a regression test using one
  of the fixture datasets under `tests/data/`.
- Public APIs need a docstring explaining inputs, outputs, and the
  reference (paper or book) for the algorithm.
- Avoid breaking the upstream `plspm` 0.5.7 API. New OpenPLS features go
  into new submodules (see `plspm/htmt.py`, `plspm/mga.py`).

## Releasing

Maintainer task. See [`docs/RELEASING.md`](docs/RELEASING.md) for the tag
+ publish flow.

## Questions

Open a GitHub issue or reach out via [openpls.app](https://openpls.app).
