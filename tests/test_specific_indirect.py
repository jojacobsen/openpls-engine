"""Specific indirect effects (mediation analysis) tests."""

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.specific_indirect import enumerate_chains


def _satisfaction_structure() -> c.Structure:
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    return structure


def _fit_satisfaction(bootstrap: bool = False) -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = _satisfaction_structure()
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
    config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
    config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")
    return Plspm(
        satisfaction,
        config,
        bootstrap=bootstrap,
        bootstrap_iterations=200,
        processes=2,
    )


def test_enumerate_chains_satisfaction():
    path = _satisfaction_structure().path()
    chains = enumerate_chains(path, "IMAG", "LOY")
    chain_set = {tuple(ch) for ch in chains}
    expected = {
        ("IMAG", "SAT", "LOY"),
        ("IMAG", "EXPE", "SAT", "LOY"),
        ("IMAG", "EXPE", "QUAL", "SAT", "LOY"),
        ("IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"),
        ("IMAG", "EXPE", "VAL", "SAT", "LOY"),
    }
    assert chain_set == expected


def test_point_estimate_matches_total_indirect():
    """Sum of all chain products equals inner-model indirect effect."""
    fit = _fit_satisfaction()
    chains_df = fit.specific_indirect_effects("IMAG", "LOY")
    effects = fit.effects()
    expected_indirect = float(effects.loc["IMAG -> LOY", "indirect"])
    actual = float(chains_df["estimate"].sum())
    npt.assert_allclose(actual, expected_indirect, atol=1e-9)


def test_point_estimate_explicit_through():
    fit = _fit_satisfaction()
    chain = fit.specific_indirect_effects("IMAG", "LOY", through=["SAT"])
    assert chain.index.tolist() == ["IMAG -> SAT -> LOY"]
    coeffs = fit.path_coefficients()
    expected = float(coeffs.loc["SAT", "IMAG"]) * float(coeffs.loc["LOY", "SAT"])
    npt.assert_allclose(float(chain.iloc[0]["estimate"]), expected, atol=1e-12)


def test_point_estimate_explicit_three_step():
    fit = _fit_satisfaction()
    chain = fit.specific_indirect_effects(
        "IMAG", "LOY", through=["EXPE", "QUAL", "SAT"]
    )
    assert chain.index.tolist() == ["IMAG -> EXPE -> QUAL -> SAT -> LOY"]
    coeffs = fit.path_coefficients()
    expected = (
        float(coeffs.loc["EXPE", "IMAG"])
        * float(coeffs.loc["QUAL", "EXPE"])
        * float(coeffs.loc["SAT", "QUAL"])
        * float(coeffs.loc["LOY", "SAT"])
    )
    npt.assert_allclose(float(chain.iloc[0]["estimate"]), expected, atol=1e-12)


def test_point_estimate_no_indirect_path_raises():
    fit = _fit_satisfaction()
    # SAT -> IMAG does not exist (and SAT is downstream of IMAG)
    with pytest.raises(ValueError, match="no indirect path"):
        fit.specific_indirect_effects("LOY", "IMAG")


def test_point_estimate_same_source_target_raises():
    fit = _fit_satisfaction()
    with pytest.raises(ValueError, match="source and target must differ"):
        fit.specific_indirect_effects("IMAG", "IMAG")


def test_point_estimate_unknown_lv_raises():
    fit = _fit_satisfaction()
    with pytest.raises(KeyError):
        fit.specific_indirect_effects("NOPE", "LOY")


def test_point_estimate_broken_through_raises():
    fit = _fit_satisfaction()
    # IMAG -> VAL does not exist as a direct edge
    with pytest.raises(ValueError, match="no direct edge"):
        fit.specific_indirect_effects("IMAG", "LOY", through=["VAL"])


def test_bootstrap_specific_indirect_has_expected_columns():
    fit = _fit_satisfaction(bootstrap=True)
    boot = fit.bootstrap().specific_indirect_effects("IMAG", "LOY")
    expected_cols = {
        "from", "to", "via", "original", "mean",
        "std.error", "perc.lower", "perc.upper", "t stat.",
    }
    assert expected_cols.issubset(set(boot.columns))
    # original column matches the point estimate
    point = fit.specific_indirect_effects("IMAG", "LOY")
    npt.assert_allclose(
        boot["original"].sort_index().to_numpy(),
        point["estimate"].sort_index().to_numpy(),
        atol=1e-12,
    )


def test_bootstrap_specific_indirect_ci_bounds_make_sense():
    fit = _fit_satisfaction(bootstrap=True)
    boot = fit.bootstrap().specific_indirect_effects(
        "IMAG", "LOY", through=["EXPE", "SAT"]
    )
    row = boot.iloc[0]
    assert row["perc.lower"] <= row["mean"] <= row["perc.upper"]
    assert row["std.error"] > 0
    # 90% CI should be tighter than 95% CI
    boot_90 = fit.bootstrap().specific_indirect_effects(
        "IMAG", "LOY", through=["EXPE", "SAT"], alpha=0.10
    )
    assert boot_90.iloc[0]["perc.lower"] >= row["perc.lower"]
    assert boot_90.iloc[0]["perc.upper"] <= row["perc.upper"]


def test_bootstrap_alpha_out_of_range_raises():
    fit = _fit_satisfaction(bootstrap=True)
    with pytest.raises(ValueError, match="alpha"):
        fit.bootstrap().specific_indirect_effects("IMAG", "LOY", alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        fit.bootstrap().specific_indirect_effects("IMAG", "LOY", alpha=1.5)
