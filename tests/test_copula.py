#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob / OpenPLS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openpls.config as c
from openpls.copula import _copula_term, _ols
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _satisfaction_plspm() -> Plspm:
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
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_copula_term_matches_phi_inv_of_empirical_cdf():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    ranks = stats.rankdata(x, method="average")
    expected = stats.norm.ppf(ranks / (len(x) + 1.0))
    np.testing.assert_allclose(_copula_term(x), expected, rtol=1e-12)


def test_copula_term_raises_when_too_few_observations():
    with pytest.raises(ValueError):
        _copula_term(np.array([1.0, 2.0]))


def test_ols_matches_numpy_lstsq_intercept_excluded():
    rng = np.random.default_rng(1)
    n = 100
    x = rng.standard_normal((n, 3))
    y = 1.0 + x @ np.array([0.5, -0.2, 0.7]) + rng.standard_normal(n) * 0.1
    got = _ols(y, x)
    design = np.column_stack([np.ones(n), x])
    beta_full, *_ = np.linalg.lstsq(design, y, rcond=None)
    np.testing.assert_allclose(got, beta_full[1:], rtol=1e-12)


def test_copula_runs_on_satisfaction_and_returns_expected_columns():
    plspm_calc = _satisfaction_plspm()
    cop = plspm_calc.copula(endogenous="SAT", n_boot=100, seed=0)
    coef = cop.coefficients()
    assert list(coef.columns) == [
        "predictor",
        "gamma",
        "boot_se",
        "t",
        "p_value",
        "cvm_p_nonnormal",
    ]
    assert list(coef["predictor"]) == ["IMAG", "EXPE", "QUAL", "VAL"]
    assert (coef["boot_se"] > 0).all()


def test_copula_endogenous_unknown_raises():
    plspm_calc = _satisfaction_plspm()
    with pytest.raises(ValueError):
        plspm_calc.copula(endogenous="DOES_NOT_EXIST")


def test_copula_suspected_not_a_predecessor_raises():
    plspm_calc = _satisfaction_plspm()
    # LOY does not predict SAT (SAT → LOY, not the other way).
    with pytest.raises(ValueError):
        plspm_calc.copula(endogenous="SAT", suspected=["LOY"])


def test_copula_exogenous_lv_raises():
    plspm_calc = _satisfaction_plspm()
    # IMAG has no predecessors, so there is nothing to test.
    with pytest.raises(ValueError):
        plspm_calc.copula(endogenous="IMAG")


def test_copula_subset_of_suspected_predictors():
    plspm_calc = _satisfaction_plspm()
    cop = plspm_calc.copula(
        endogenous="SAT", suspected=["IMAG", "QUAL"], n_boot=100, seed=0
    )
    assert cop.suspected() == ["IMAG", "QUAL"]
    assert cop.predictors() == ["IMAG", "EXPE", "QUAL", "VAL"]
    # augmented_paths still spans all predictors.
    assert list(cop.augmented_paths().index) == ["IMAG", "EXPE", "QUAL", "VAL"]
    assert list(cop.coefficients()["predictor"]) == ["IMAG", "QUAL"]


def test_copula_summary_marks_normal_predictors_inadmissible():
    """A Gaussian regressor must trigger ``copula not admissible (normal)``
    because Phi^{-1}(F_n(X)) collapses to X itself under normality, and
    the test cannot tell endogeneity from a normal predictor."""
    rng = np.random.default_rng(42)
    n = 500
    eta_x = rng.standard_normal(n)
    eta_y = 0.5 * eta_x + rng.standard_normal(n) * np.sqrt(1 - 0.25)
    loading = 0.9
    p = 4
    err_x = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    err_y = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    df = pd.DataFrame(
        np.column_stack([
            loading * eta_x[:, None] + err_x,
            loading * eta_y[:, None] + err_y,
        ]),
        columns=[f"x{i+1}" for i in range(p)] + [f"y{i+1}" for i in range(p)],
    )
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("X", Mode.A, df, "x")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")
    plspm_calc = Plspm(df, config, Scheme.CENTROID)
    summary = plspm_calc.copula(endogenous="Y", n_boot=100, seed=0).summary()
    decision = summary.loc[summary["predictor"] == "X", "decision"].iloc[0]
    assert decision == "copula not admissible (normal)"


def test_copula_detects_endogeneity_with_non_normal_endogenous_regressor():
    """Build a structural model where an endogenous, *skewed* predictor X
    is correlated with the omitted-variable error in Y. The Park-Gupta
    copula coefficient should reject H0: gamma = 0."""
    rng = np.random.default_rng(123)
    n = 600
    # Common omitted confound z (skewed via squared standard normal).
    z = rng.standard_normal(n) ** 2
    # Skewed endogenous predictor x = z + skewed noise (chi-square-like).
    x_score = z + rng.exponential(scale=1.0, size=n)
    # y depends on x AND on the omitted confound z → endogeneity.
    y_score = 0.5 * x_score + 0.8 * z + rng.standard_normal(n)
    loading = 0.9
    p = 4
    err_x = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    err_y = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    df = pd.DataFrame(
        np.column_stack([
            loading * x_score[:, None] + err_x,
            loading * y_score[:, None] + err_y,
        ]),
        columns=[f"x{i+1}" for i in range(p)] + [f"y{i+1}" for i in range(p)],
    )
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("X", Mode.A, df, "x")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")
    plspm_calc = Plspm(df, config, Scheme.CENTROID)
    cop = plspm_calc.copula(endogenous="Y", n_boot=500, seed=0)
    summary = cop.summary()
    row = summary.iloc[0]
    # Non-normality must be detected (admissibility) and gamma must be
    # significant at 5 %.
    assert row["cvm_p_nonnormal"] < 0.05
    assert row["p_value"] < 0.05
    assert row["decision"] == "endogeneity detected"


def test_copula_n_boot_minimum_enforced():
    plspm_calc = _satisfaction_plspm()
    with pytest.raises(ValueError):
        plspm_calc.copula(endogenous="SAT", n_boot=10)


def test_copula_deterministic_under_fixed_seed():
    plspm_calc = _satisfaction_plspm()
    a = plspm_calc.copula(endogenous="SAT", n_boot=200, seed=7).coefficients()
    b = plspm_calc.copula(endogenous="SAT", n_boot=200, seed=7).coefficients()
    pd.testing.assert_frame_equal(a, b)


def test_copula_augmented_paths_differ_from_raw_when_endogeneity_present():
    rng = np.random.default_rng(321)
    n = 600
    z = rng.standard_normal(n) ** 2
    x_score = z + rng.exponential(scale=1.0, size=n)
    y_score = 0.5 * x_score + 0.8 * z + rng.standard_normal(n)
    loading = 0.9
    p = 4
    err_x = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    err_y = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    df = pd.DataFrame(
        np.column_stack([
            loading * x_score[:, None] + err_x,
            loading * y_score[:, None] + err_y,
        ]),
        columns=[f"x{i+1}" for i in range(p)] + [f"y{i+1}" for i in range(p)],
    )
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("X", Mode.A, df, "x")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")
    plspm_calc = Plspm(df, config, Scheme.CENTROID)
    raw = float(plspm_calc.path_coefficients().loc["Y", "X"])
    corrected = float(
        plspm_calc.copula(endogenous="Y", n_boot=200, seed=0)
        .augmented_paths()
        .loc["X"]
    )
    # Endogeneity correction must move the structural estimate noticeably.
    assert abs(raw - corrected) > 1e-2
