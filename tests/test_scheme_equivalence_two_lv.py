"""Regression test: in a two-LV model the inner-weighting scheme is
structurally degenerate. Each LV has exactly one neighbour, so the
inner-weights update collapses to the trivial correlation and PATH,
CENTROID, FACTORIAL, PCA, and NEWTON must all produce identical
path coefficients, R², loadings, and weights.

Motivation: a long-running investigation traced an apparent scheme-related
drift to a reference-data mismatch, not an engine bug. Locking down the
two-LV equivalence as a regression test prevents future scheme changes
from silently breaking this invariant."""

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _two_lv_formative_to_reflective(n: int = 400, seed: int = 1) -> tuple[pd.DataFrame, c.Config]:
    rng = np.random.default_rng(seed)
    # Three correlated formative drivers feeding into one reflective LV
    # whose three indicators load on a common factor.
    common_driver = rng.standard_normal(n)
    drv = np.column_stack([
        0.7 * common_driver + 0.3 * rng.standard_normal(n) for _ in range(3)
    ])
    target_factor = 0.6 * drv.sum(axis=1) / np.sqrt(3) + 0.4 * rng.standard_normal(n)
    out = np.column_stack([
        0.8 * target_factor + 0.2 * rng.standard_normal(n) for _ in range(3)
    ])
    data = pd.DataFrame(
        np.column_stack([drv, out]),
        columns=["drv_1", "drv_2", "drv_3", "out_1", "out_2", "out_3"],
    )
    structure = c.Structure()
    structure.add_path(["DRV"], ["OUT"])
    config = c.Config(structure.path(), scaled=True)
    config.add_lv("DRV", Mode.B, c.MV("drv_1"), c.MV("drv_2"), c.MV("drv_3"))
    config.add_lv("OUT", Mode.A, c.MV("out_1"), c.MV("out_2"), c.MV("out_3"))
    return data, config


def _fit(scheme: Scheme):
    data, config = _two_lv_formative_to_reflective()
    return Plspm(data, config, scheme)


def test_all_inner_schemes_give_identical_paths_on_two_lv_model():
    schemes = [Scheme.PATH, Scheme.CENTROID, Scheme.FACTORIAL, Scheme.PCA, Scheme.NEWTON]
    fits = [_fit(s) for s in schemes]
    reference = fits[0].path_coefficients().values
    for s, fit in zip(schemes[1:], fits[1:]):
        diff = float(np.max(np.abs(fit.path_coefficients().values - reference)))
        assert diff < 1e-6, f"scheme={s} produced different path coefficients (max |Δ|={diff})"


def test_all_inner_schemes_give_identical_outer_model_on_two_lv_model():
    schemes = [Scheme.PATH, Scheme.CENTROID, Scheme.FACTORIAL, Scheme.PCA, Scheme.NEWTON]
    fits = [_fit(s) for s in schemes]
    ref_outer = fits[0].outer_model()
    for s, fit in zip(schemes[1:], fits[1:]):
        outer = fit.outer_model()
        for col in ["weight", "loading"]:
            diff = float(np.max(np.abs(outer[col].values - ref_outer[col].values)))
            assert diff < 1e-6, f"scheme={s} {col} differs (max |Δ|={diff})"


def test_all_inner_schemes_give_identical_r_squared_on_two_lv_model():
    schemes = [Scheme.PATH, Scheme.CENTROID, Scheme.FACTORIAL, Scheme.PCA, Scheme.NEWTON]
    fits = [_fit(s) for s in schemes]
    ref_r2 = fits[0].inner_summary()["r_squared"].values
    for s, fit in zip(schemes[1:], fits[1:]):
        diff = float(np.max(np.abs(fit.inner_summary()["r_squared"].values - ref_r2)))
        assert diff < 1e-6, f"scheme={s} R² differs (max |Δ|={diff})"
