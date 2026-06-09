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

import openpls.config as c
from openpls.config import MV
from openpls.mode import Mode
from openpls.plsc import _rho_a_block
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _satisfaction_plspm(mode: Mode = Mode.A) -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, mode, satisfaction, lv.lower())
    return Plspm(satisfaction, config, Scheme.CENTROID)


def _one_factor_block(n: int, p: int, loading: float, seed: int) -> pd.DataFrame:
    """Generate ``p`` indicators of a single latent factor with the given
    standardized loading and the matching error variance ``1 - loading²``."""
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(n)
    err = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    data = loading * eta[:, None] + err
    return pd.DataFrame(data, columns=[f"x{i+1}" for i in range(p)])


def test_rho_a_block_matches_dijkstra_henseler_formula():
    """rho_A on a deterministic block must equal the closed-form
    ``(w'w)² * w'Sw / w'(ww' - diag) w`` (Dijkstra & Henseler 2015)."""
    rng = np.random.default_rng(0)
    block = rng.standard_normal((200, 4)) + rng.standard_normal(200)[:, None]
    w = np.array([0.3, 0.35, 0.25, 0.4])
    got = _rho_a_block(w, block)

    z = (block - block.mean(axis=0)) / block.std(axis=0, ddof=1)
    s = np.cov(z, rowvar=False, ddof=1)
    np.fill_diagonal(s, 0.0)
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0.0)
    expected = (float(w @ w) ** 2) * float(w @ s @ w) / float(w @ ww @ w)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_rho_a_block_single_indicator_returns_one():
    assert _rho_a_block(np.array([1.0]), np.array([[1.0], [2.0], [3.0]])) == 1.0


def _fit_one_factor(n: int, p: int, loading: float, seed: int) -> Plspm:
    """Fit a trivial one-LV PLS model (Y ← X both 5-indicator factors)
    so the PLS-normalized outer weights are available."""
    rng = np.random.default_rng(seed)
    eta_x = rng.standard_normal(n)
    eta_y = 0.5 * eta_x + rng.standard_normal(n) * np.sqrt(1 - 0.25)
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
    return Plspm(df, config, Scheme.CENTROID)


def test_rho_a_higher_when_loadings_are_higher():
    """rho_A should be larger for a block of strongly-loading indicators
    than for a noisier block, given PLS-normalized outer weights."""
    rho_high = _fit_one_factor(n=1000, p=5, loading=0.9, seed=1).plsc().rho_a().loc["X"]
    rho_low = _fit_one_factor(n=1000, p=5, loading=0.4, seed=1).plsc().rho_a().loc["X"]
    assert rho_low < rho_high
    assert 0.85 < rho_high < 1.05  # near 1 for strong loadings under PLS weights


def test_plsc_mode_b_blocks_get_rho_one():
    """Formative (Mode B) constructs are not common-factor measurements,
    so PLSc by definition leaves them with rho_A = 1."""
    plspm_calc = _satisfaction_plspm(mode=Mode.B)
    rho = plspm_calc.plsc().rho_a()
    np.testing.assert_allclose(rho.values, 1.0)


def test_plsc_single_indicator_lv_gets_rho_one():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv("EXPE", Mode.A, MV("expe1"))
    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    rho = plspm_calc.plsc().rho_a()
    assert rho.loc["EXPE"] == 1.0
    assert rho.loc["IMAG"] != 1.0  # multi-indicator → real rho_A


def test_plsc_corrected_paths_match_ols_on_adjusted_correlations():
    """The corrected path coefficients should equal the OLS solution on
    the dis-attenuated construct correlation matrix."""
    plspm_calc = _satisfaction_plspm()
    plsc = plspm_calc.plsc()
    adj = plsc.adjusted_correlations().to_numpy()
    lvs = list(plspm_calc.path_coefficients().columns)
    paths = plsc.path_coefficients()
    # Recompute SAT ← IMAG + EXPE + QUAL + VAL by hand.
    ix = lambda name: lvs.index(name)
    pred = [ix("IMAG"), ix("EXPE"), ix("QUAL"), ix("VAL")]
    r_xx = adj[np.ix_(pred, pred)]
    r_xy = adj[pred, ix("SAT")]
    expected = np.linalg.solve(r_xx, r_xy)
    np.testing.assert_allclose(
        paths.loc["SAT", ["IMAG", "EXPE", "QUAL", "VAL"]].to_numpy(),
        expected,
        rtol=1e-10,
    )


def test_plsc_corrected_paths_inflate_under_attenuation():
    """When reliabilities are below 1, the corrected (dis-attenuated)
    path coefficient magnitudes should be ≥ the raw ones for simple
    chains (a → b)."""
    n, p = 1000, 4
    rng = np.random.default_rng(7)
    eta_x = rng.standard_normal(n)
    eta_y = 0.5 * eta_x + rng.standard_normal(n) * np.sqrt(1 - 0.5 ** 2)
    loading = 0.7
    x_inds = loading * eta_x[:, None] + rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    y_inds = loading * eta_y[:, None] + rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
    df = pd.DataFrame(
        np.column_stack([x_inds, y_inds]),
        columns=[f"x{i+1}" for i in range(p)] + [f"y{i+1}" for i in range(p)],
    )
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("X", Mode.A, df, "x")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")
    plspm_calc = Plspm(df, config, Scheme.CENTROID)

    raw_path = float(plspm_calc.path_coefficients().loc["Y", "X"])
    corr_path = float(plspm_calc.plsc().path_coefficients().loc["Y", "X"])
    rho = plspm_calc.plsc().rho_a()
    assert rho.loc["X"] < 1.0 or rho.loc["Y"] < 1.0
    assert abs(corr_path) >= abs(raw_path) - 1e-9


def test_plsc_corrected_loadings_match_consistent_factor_formula():
    """For Mode A blocks: corrected loading = w_k * sqrt(rho_A) / (w'w)."""
    plspm_calc = _satisfaction_plspm()
    plsc = plspm_calc.plsc()
    outer = plspm_calc.outer_model()
    rho = plsc.rho_a()
    corrected = plsc.loadings()
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        inds = [c_ for c_ in outer.index if outer.loc[c_].notna().all()]  # noqa
        # Restrict to indicators of this LV by matching weight presence.
        block = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
        cols = [c_ for c_ in block.columns if c_.startswith(lv.lower())]
        w = outer.loc[cols, "weight"].to_numpy(dtype=float)
        wTw = float(w @ w)
        expected = w * np.sqrt(rho.loc[lv]) / wTw
        np.testing.assert_allclose(corrected.loc[cols].to_numpy(), expected, rtol=1e-10)


def test_plsc_summary_columns_and_shape():
    plspm_calc = _satisfaction_plspm()
    summary = plspm_calc.plsc().summary()
    assert list(summary.columns) == ["rho_a", "r_squared", "r_squared_adj"]
    assert list(summary.index) == ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]
    # IMAG is exogenous; R² should be NaN.
    assert np.isnan(summary.loc["IMAG", "r_squared"])
    assert np.isnan(summary.loc["IMAG", "r_squared_adj"])
    # Endogenous LVs should have non-NaN R².
    for endo in ["EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        assert not np.isnan(summary.loc[endo, "r_squared"])


def test_plsc_is_cached_across_calls():
    plspm_calc = _satisfaction_plspm()
    a = plspm_calc.plsc()
    b = plspm_calc.plsc()
    assert a is b


def test_plsc_deterministic_across_fits():
    plspm_calc_a = _satisfaction_plspm()
    plspm_calc_b = _satisfaction_plspm()
    pd.testing.assert_series_equal(
        plspm_calc_a.plsc().rho_a(),
        plspm_calc_b.plsc().rho_a(),
    )
    pd.testing.assert_frame_equal(
        plspm_calc_a.plsc().path_coefficients(),
        plspm_calc_b.plsc().path_coefficients(),
    )
