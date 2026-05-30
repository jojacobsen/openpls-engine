import math

import numpy as np
import pandas as pd

import plspm.config as c
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scheme import Scheme


def _satisfaction():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return satisfaction, config


def test_newton_scheme_converges_on_satisfaction():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.NEWTON)
    paths = fit.path_coefficients()
    assert paths.shape == (6, 6)
    assert np.isfinite(paths.values).all()


def test_newton_scheme_path_coefficients_close_to_path_scheme():
    data, config = _satisfaction()
    p_path = Plspm(data, config, Scheme.PATH).path_coefficients()
    p_newton = Plspm(data, config, Scheme.NEWTON).path_coefficients()
    # NEWTON jointly fits OLS over both predecessors and successors, where
    # PATH uses correlations for successors. Path estimates should be
    # similar but not identical.
    diff = (p_path - p_newton).abs().max().max()
    assert diff < 0.05
    assert diff > 1e-6  # not literally identical


def test_newton_scheme_r_squared_is_reasonable():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.NEWTON)
    summary = fit.inner_summary()
    # SAT in the ECSI satisfaction model has R² well above 0.6 under any
    # sensible inner-weighting scheme.
    sat_r2 = float(summary.loc["SAT", "r_squared"])
    assert sat_r2 > 0.6
    assert sat_r2 < 1.0


def test_newton_scheme_produces_scores_of_correct_shape():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.NEWTON)
    scores = fit.scores()
    # 250 observations × 6 LVs in the satisfaction dataset.
    assert scores.shape == (250, 6)
    assert set(scores.columns) == {"IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"}


def test_newton_scheme_is_deterministic():
    data, config = _satisfaction()
    p1 = Plspm(data, config, Scheme.NEWTON).path_coefficients().values
    p2 = Plspm(data, config, Scheme.NEWTON).path_coefficients().values
    assert np.allclose(p1, p2, atol=1e-9)


def test_newton_scheme_outer_loadings_close_to_path_scheme():
    data, config = _satisfaction()
    o_path = Plspm(data, config, Scheme.PATH).outer_model()["loading"]
    o_newton = Plspm(data, config, Scheme.NEWTON).outer_model()["loading"]
    diff = (o_path - o_newton).abs().max()
    assert diff < 0.05


def test_newton_scheme_works_with_bootstrap():
    data, config = _satisfaction()
    fit = Plspm(
        data, config, Scheme.NEWTON,
        bootstrap=True, bootstrap_iterations=100, processes=2,
    )
    boot = fit.bootstrap()
    # path coefficient point-estimates exist for the bootstrap
    paths = boot.paths()
    assert "original" in paths.columns
    assert math.isfinite(float(paths["original"].iloc[0]))
