"""Cohen f² effect-size tests."""

import math

import numpy.testing as npt
import pandas as pd
import pytest
import statsmodels.api as sm

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm


def _fit_satisfaction() -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
    config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
    config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")
    return Plspm(satisfaction, config)


def test_f_squared_table_has_row_per_structural_edge():
    fit = _fit_satisfaction()
    table = fit.f_squared().table()
    structure_edges = {
        "IMAG -> EXPE", "IMAG -> SAT", "IMAG -> LOY",
        "EXPE -> QUAL", "EXPE -> VAL", "EXPE -> SAT",
        "QUAL -> VAL", "QUAL -> SAT",
        "VAL -> SAT",
        "SAT -> LOY",
    }
    assert set(table.index) == structure_edges
    expected_cols = {
        "from", "to", "r_squared_full", "r_squared_reduced",
        "f_squared", "effect_size",
    }
    assert expected_cols.issubset(set(table.columns))


def test_f_squared_matches_manual_refit():
    """f² from the API equals the value computed by manual OLS refit."""
    fit = _fit_satisfaction()
    scores = fit.scores()
    r2_full = float(fit.inner_summary().loc["SAT", "r_squared"])
    # SAT has predictors IMAG, EXPE, QUAL, VAL — drop EXPE and refit.
    reduced = ["IMAG", "QUAL", "VAL"]
    exog = sm.add_constant(scores.loc[:, reduced])
    r2_red = float(sm.OLS(scores.loc[:, "SAT"], exog).fit().rsquared)
    expected = (r2_full - r2_red) / (1.0 - r2_full)
    actual = float(fit.f_squared().table().loc["EXPE -> SAT", "f_squared"])
    npt.assert_allclose(actual, expected, atol=1e-10)


def test_f_squared_single_predictor_equals_r2_over_one_minus_r2():
    """For an endogenous LV with one predictor, f² = R²/(1−R²) (reduced R² = 0)."""
    fit = _fit_satisfaction()
    # LOY has predictors IMAG and SAT — there is no single-predictor LV in
    # the satisfaction model, so build the identity from QUAL (only EXPE)
    r2_qual = float(fit.inner_summary().loc["QUAL", "r_squared"])
    expected = r2_qual / (1.0 - r2_qual)
    actual = float(fit.f_squared().table().loc["EXPE -> QUAL", "f_squared"])
    npt.assert_allclose(actual, expected, atol=1e-10)


def test_f_squared_effect_size_labels():
    fit = _fit_satisfaction()
    table = fit.f_squared().table()
    for label, value in zip(table["effect_size"], table["f_squared"]):
        if value < 0.02:
            assert label == "none"
        elif value < 0.15:
            assert label == "small"
        elif value < 0.35:
            assert label == "medium"
        else:
            assert label == "large"


def test_f_squared_matrix_matches_table():
    fit = _fit_satisfaction()
    table = fit.f_squared().table()
    matrix = fit.f_squared().matrix()
    for _, row in table.iterrows():
        npt.assert_allclose(
            matrix.loc[row["to"], row["from"]], row["f_squared"], atol=1e-12
        )
    # cells with no edge are NaN
    assert math.isnan(matrix.loc["IMAG", "SAT"])


def test_f_squared_is_cached():
    fit = _fit_satisfaction()
    assert fit.f_squared() is fit.f_squared()
