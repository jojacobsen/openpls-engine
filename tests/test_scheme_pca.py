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


def test_pca_scheme_converges_on_satisfaction():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.PCA)
    paths = fit.path_coefficients()
    assert paths.shape == (6, 6)
    assert np.isfinite(paths.values).all()


def test_pca_scheme_path_coefficients_close_to_path_scheme():
    data, config = _satisfaction()
    p_path = Plspm(data, config, Scheme.PATH).path_coefficients()
    p_pca = Plspm(data, config, Scheme.PCA).path_coefficients()
    diff = (p_path - p_pca).abs().max().max()
    assert diff < 0.05
    assert diff > 1e-6


def test_pca_scheme_r_squared_is_reasonable():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.PCA)
    summary = fit.inner_summary()
    sat_r2 = float(summary.loc["SAT", "r_squared"])
    assert sat_r2 > 0.6
    assert sat_r2 < 1.0


def test_pca_scheme_is_deterministic():
    data, config = _satisfaction()
    p1 = Plspm(data, config, Scheme.PCA).path_coefficients().values
    p2 = Plspm(data, config, Scheme.PCA).path_coefficients().values
    assert np.allclose(p1, p2, atol=1e-9)


def test_pca_scheme_produces_scores_of_correct_shape():
    data, config = _satisfaction()
    fit = Plspm(data, config, Scheme.PCA)
    scores = fit.scores()
    assert scores.shape == (250, 6)
    assert set(scores.columns) == {"IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"}


def test_pca_scheme_outer_loadings_close_to_path_scheme():
    data, config = _satisfaction()
    o_path = Plspm(data, config, Scheme.PATH).outer_model()["loading"]
    o_pca = Plspm(data, config, Scheme.PCA).outer_model()["loading"]
    diff = (o_path - o_pca).abs().max()
    assert diff < 0.05


def test_pca_scheme_works_with_bootstrap():
    data, config = _satisfaction()
    fit = Plspm(
        data, config, Scheme.PCA,
        bootstrap=True, bootstrap_iterations=100, processes=2,
    )
    boot = fit.bootstrap()
    paths = boot.paths()
    assert "original" in paths.columns
    assert np.isfinite(float(paths["original"].iloc[0]))
