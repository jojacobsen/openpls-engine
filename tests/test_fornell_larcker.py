"""Fornell-Larcker discriminant-validity tests."""

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm


def _fit_satisfaction(modes: dict[str, Mode] | None = None) -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    modes = modes or {}
    for lv, prefix in [("IMAG", "imag"), ("EXPE", "expe"), ("QUAL", "qual"),
                       ("VAL", "val"), ("SAT", "sat"), ("LOY", "loy")]:
        config.add_lv_with_columns_named(
            lv, modes.get(lv, Mode.A), satisfaction, prefix
        )
    return Plspm(satisfaction, config)


def test_fornell_larcker_matrix_shape():
    fit = _fit_satisfaction()
    matrix = fit.fornell_larcker().matrix()
    expected_lvs = ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]
    assert sorted(matrix.index.tolist()) == sorted(expected_lvs)
    assert sorted(matrix.columns.tolist()) == sorted(expected_lvs)


def test_fornell_larcker_diagonal_is_sqrt_ave():
    fit = _fit_satisfaction()
    fl = fit.fornell_larcker()
    summary = fit.inner_summary()
    for lv in fl.matrix().index:
        ave = float(summary.loc[lv, "ave"])
        npt.assert_allclose(
            fl.matrix().loc[lv, lv], math.sqrt(ave), atol=1e-12
        )


def test_fornell_larcker_off_diagonal_matches_scores_correlation():
    fit = _fit_satisfaction()
    fl = fit.fornell_larcker()
    corr = fit.scores().corr()
    for a in fl.matrix().index:
        for b in fl.matrix().columns:
            if a == b:
                continue
            npt.assert_allclose(
                fl.matrix().loc[a, b], corr.loc[a, b], atol=1e-12
            )


def test_fornell_larcker_matrix_is_symmetric():
    fit = _fit_satisfaction()
    matrix = fit.fornell_larcker().matrix()
    npt.assert_allclose(matrix.values, matrix.values.T, atol=1e-12)


def test_fornell_larcker_summary_columns():
    fit = _fit_satisfaction()
    summary = fit.fornell_larcker().summary()
    expected_cols = {"sqrt_ave", "max_abs_corr", "passes", "note"}
    assert expected_cols.issubset(set(summary.columns))


def test_fornell_larcker_summary_passes_logic():
    """``passes`` is True iff sqrt(AVE) > max |off-diagonal|."""
    fit = _fit_satisfaction()
    summary = fit.fornell_larcker().summary()
    for lv, row in summary.iterrows():
        if pd.isna(row["passes"]):
            continue
        expected = bool(row["sqrt_ave"] > row["max_abs_corr"])
        assert bool(row["passes"]) is expected


def test_fornell_larcker_mode_b_has_nan_diagonal():
    """Mode-B LVs have undefined AVE and should be NaN on the diagonal."""
    fit = _fit_satisfaction(modes={"IMAG": Mode.B})
    fl = fit.fornell_larcker()
    assert math.isnan(fl.matrix().loc["IMAG", "IMAG"])
    assert math.isnan(fl.ave().loc["IMAG"])
    # the row for IMAG in summary should signal undefined
    assert pd.isna(fl.summary().loc["IMAG", "passes"])
    assert "no AVE" in fl.summary().loc["IMAG", "note"]


def test_fornell_larcker_is_cached():
    fit = _fit_satisfaction()
    assert fit.fornell_larcker() is fit.fornell_larcker()
