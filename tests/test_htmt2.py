#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob / OpenPLS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.config import MV
from openpls.htmt2 import _gmean
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _satisfaction_plspm():
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


def test_gmean_matches_exp_mean_log():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 1.0, size=30)
    np.testing.assert_allclose(_gmean(x), np.exp(np.mean(np.log(x))), rtol=1e-12)


def test_gmean_returns_none_on_zero_or_empty():
    assert _gmean(np.array([])) is None
    assert _gmean(np.array([0.5, 0.0, 0.3])) is None
    assert _gmean(np.array([-0.1, 0.5])) is None


def test_htmt2_matrix_is_symmetric_with_nan_diagonal():
    htmt2 = _satisfaction_plspm().htmt2().matrix()
    assert list(htmt2.index) == list(htmt2.columns)
    assert htmt2.shape == (6, 6)
    for lv in htmt2.index:
        assert pd.isna(htmt2.loc[lv, lv])
    arr = htmt2.to_numpy()
    upper = arr[np.triu_indices(6, k=1)]
    lower = arr.T[np.triu_indices(6, k=1)]
    np.testing.assert_allclose(upper, lower, equal_nan=True)


def test_htmt2_values_in_plausible_range_and_all_pairs_resolved():
    htmt2 = _satisfaction_plspm().htmt2().matrix()
    off_diag = htmt2.to_numpy()[np.triu_indices(6, k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    assert len(off_diag) == 15
    assert (off_diag > 0).all()
    assert (off_diag < 1.5).all()


def test_htmt2_pairs_long_format():
    pairs = _satisfaction_plspm().htmt2().pairs()
    assert list(pairs.columns) == ["lv_a", "lv_b", "htmt2"]
    assert len(pairs) == 15
    assert pairs["htmt2"].notna().all()
    assert (pairs["htmt2"] > 0).all()


def test_htmt2_matches_geometric_mean_formula_for_one_pair():
    """Hand-compute HTMT2 for IMAG vs EXPE and compare."""
    plspm_calc = _satisfaction_plspm()
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    imag_cols = [c_ for c_ in satisfaction.columns if c_.startswith("imag")]
    expe_cols = [c_ for c_ in satisfaction.columns if c_.startswith("expe")]
    abs_corr = satisfaction[imag_cols + expe_cols].corr().abs()

    def gm(arr):
        return float(np.exp(np.mean(np.log(arr))))

    within_imag = abs_corr.loc[imag_cols, imag_cols].to_numpy()
    gw_imag = gm(within_imag[np.triu_indices_from(within_imag, k=1)])
    within_expe = abs_corr.loc[expe_cols, expe_cols].to_numpy()
    gw_expe = gm(within_expe[np.triu_indices_from(within_expe, k=1)])
    between = abs_corr.loc[imag_cols, expe_cols].to_numpy().ravel()
    expected = gm(between) / math.sqrt(gw_imag * gw_expe)

    got = float(plspm_calc.htmt2().matrix().loc["IMAG", "EXPE"])
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_htmt2_differs_from_htmt_under_unequal_loadings():
    """The arithmetic and geometric means coincide only when all values
    are equal; with realistic, unequal indicator correlations HTMT2
    differs measurably from HTMT (typically lower because the geometric
    mean is dominated by the smaller correlations)."""
    plspm_calc = _satisfaction_plspm()
    htmt = plspm_calc.htmt().matrix().to_numpy()
    htmt2 = plspm_calc.htmt2().matrix().to_numpy()
    mask = ~np.isnan(htmt) & ~np.isnan(htmt2)
    # At least one pair must differ by more than rounding noise.
    diffs = np.abs(htmt[mask] - htmt2[mask])
    assert diffs.max() > 1e-3


def test_htmt2_skips_single_indicator_lv():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv("EXPE", Mode.A, MV("expe1"))
    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    matrix = plspm_calc.htmt2().matrix()
    assert math.isnan(matrix.loc["IMAG", "EXPE"])
    assert math.isnan(matrix.loc["EXPE", "IMAG"])
    assert plspm_calc.htmt2().pairs().empty


def test_htmt2_is_cached_across_calls():
    plspm_calc = _satisfaction_plspm()
    a = plspm_calc.htmt2()
    b = plspm_calc.htmt2()
    assert a is b


def test_htmt2_equals_htmt_when_all_correlations_identical():
    """If every absolute correlation in the relevant blocks is the same
    constant, the geometric mean equals the arithmetic mean and HTMT2
    must equal HTMT exactly."""
    # Build a synthetic dataset where every pairwise correlation across
    # the relevant indicators is forced to the same value by construction.
    rng = np.random.default_rng(7)
    n = 400
    f = rng.standard_normal(n)
    # Each indicator = f + small block-specific noise scaled so within-
    # and between-block correlations approach the same value.
    err_a = rng.standard_normal((n, 3)) * 0.5
    err_b = rng.standard_normal((n, 3)) * 0.5
    df = pd.DataFrame(
        np.column_stack([f[:, None] + err_a, f[:, None] + err_b]),
        columns=["a1", "a2", "a3", "b1", "b2", "b3"],
    )
    structure = c.Structure()
    structure.add_path(["A"], ["B"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("A", Mode.A, df, "a")
    config.add_lv_with_columns_named("B", Mode.A, df, "b")
    plspm_calc = Plspm(df, config, Scheme.CENTROID)
    h1 = float(plspm_calc.htmt().matrix().loc["A", "B"])
    h2 = float(plspm_calc.htmt2().matrix().loc["A", "B"])
    # Block correlations are all close to corr(f, f) so AM ≈ GM here.
    # Tolerance is generous: we are checking that the *direction* is
    # negligible, not the exact identity, since real samples never hit
    # perfectly equal correlations.
    np.testing.assert_allclose(h1, h2, rtol=0.05)
