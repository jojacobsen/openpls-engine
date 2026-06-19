"""Publication-ready Report tests."""

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.report import Report


def _fit_satisfaction() -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    for lv, prefix in [("IMAG", "imag"), ("EXPE", "expe"), ("QUAL", "qual"),
                       ("VAL", "val"), ("SAT", "sat"), ("LOY", "loy")]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, prefix)
    return Plspm(satisfaction, config)


def test_report_is_instance():
    fit = _fit_satisfaction()
    assert isinstance(fit.report(), Report)


def test_reliability_columns_and_index():
    fit = _fit_satisfaction()
    rel = fit.report().reliability()
    expected_cols = {"mode", "mvs", "cronbach_alpha", "rho_a", "rho_c", "ave"}
    assert expected_cols.issubset(set(rel.columns))
    assert sorted(rel.index.tolist()) == sorted(
        ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]
    )


def test_reliability_without_rho_a():
    fit = _fit_satisfaction()
    rel = fit.report(include_rho_a=False).reliability()
    assert "rho_a" not in rel.columns


def test_reliability_values_match_underlying_methods():
    fit = _fit_satisfaction()
    report_rel = fit.report().reliability()
    rel = fit.reliability()
    inner = fit.inner_summary()
    for lv in report_rel.index:
        npt.assert_allclose(
            report_rel.loc[lv, "cronbach_alpha"], rel.loc[lv, "cronbach_alpha"], atol=1e-12
        )
        npt.assert_allclose(
            report_rel.loc[lv, "rho_c"], rel.loc[lv, "dillon_goldstein_rho"], atol=1e-12
        )
        npt.assert_allclose(report_rel.loc[lv, "ave"], inner.loc[lv, "ave"], atol=1e-12)


def test_discriminant_validity_keys_default():
    fit = _fit_satisfaction()
    dv = fit.report().discriminant_validity()
    assert set(dv.keys()) >= {
        "htmt", "htmt_pairs", "htmt2", "htmt2_pairs",
        "fornell_larcker", "fornell_larcker_summary",
    }


def test_discriminant_validity_without_htmt2():
    fit = _fit_satisfaction()
    dv = fit.report(include_htmt2=False).discriminant_validity()
    assert "htmt2" not in dv
    assert "htmt2_pairs" not in dv
    assert "fornell_larcker" in dv


def test_paths_columns_and_f_squared_joined():
    fit = _fit_satisfaction()
    paths = fit.report().paths()
    expected = ["from", "to", "estimate", "std_error", "t", "p_value",
                "f_squared", "effect_size"]
    assert list(paths.columns) == expected
    # f_squared values must match the FSquared table for every path
    f2 = fit.f_squared().table()
    for path in paths.index:
        npt.assert_allclose(
            paths.loc[path, "f_squared"], f2.loc[path, "f_squared"], atol=1e-12
        )
        assert paths.loc[path, "effect_size"] == f2.loc[path, "effect_size"]


def test_construct_summary_columns():
    fit = _fit_satisfaction()
    cs = fit.report().construct_summary()
    expected = ["type", "mvs", "r_squared", "r_squared_adj", "bic",
                "block_communality", "mean_redundancy"]
    assert list(cs.columns) == expected


def test_fit_indices_keys_and_finite():
    fit = _fit_satisfaction()
    f = fit.report().fit_indices()
    assert set(f.index) == {"srmr", "d_uls", "goodness_of_fit"}
    assert np.isfinite(f["srmr"])
    assert np.isfinite(f["d_uls"])
    # goodness_of_fit may be NaN if all blocks are single-item; here it should be finite
    assert np.isfinite(f["goodness_of_fit"])


def test_collinearity_keys():
    fit = _fit_satisfaction()
    col = fit.report().collinearity()
    assert set(col.keys()) == {"items", "inner"}


def test_to_dict_bundles_all_sections():
    fit = _fit_satisfaction()
    bundle = fit.report().to_dict()
    assert set(bundle.keys()) == {
        "reliability", "discriminant_validity", "paths",
        "construct_summary", "fit_indices", "collinearity",
    }


def test_repr_contains_fit_summary():
    fit = _fit_satisfaction()
    rep = repr(fit.report())
    assert "openpls.Report" in rep
    assert "SRMR=" in rep
    assert "d_ULS=" in rep
