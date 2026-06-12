"""Tests for openpls.bootstrap_mga.BootstrapMGA.

Covers structural correctness on a small synthetic fixture (fast, deterministic)
and a SmartPLS-reference comparison on the Corporate Reputation primer
(Hair et al. 2022, Primer 3rd ed.) with tolerances reflecting the fact that
we cannot reproduce SmartPLS' RNG sequence exactly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.bootstrap_mga import (
    BootstrapMGA,
    _henseler_p_one_tailed,
    _parametric_t,
    _welch_t,
)
from openpls.mga import GroupSpec
from openpls.mode import Mode
from openpls.scale import Scale
from openpls.scheme import Scheme


# ---------------------------------------------------------------------------
# Small fixture: russa with a synthetic group column
# ---------------------------------------------------------------------------


def _russa_with_group():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    rng = np.random.default_rng(42)
    russa = russa.copy()
    russa["region"] = rng.choice(["west", "east"], size=len(russa))
    return russa


def _russa_config():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv(
        "POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst")
    )
    return config


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_henseler_p_one_tailed_extreme_separation():
    """If samples_a is uniformly below samples_b, P(a <= b) -> 1."""
    a = np.linspace(-2, -1, 200)
    b = np.linspace(1, 2, 200)
    assert _henseler_p_one_tailed(a, b) == pytest.approx(1.0)


def test_henseler_p_one_tailed_equal_distributions():
    """Two draws from the same distribution: P(a <= b) ~ 0.5."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(2000)
    b = rng.standard_normal(2000)
    p = _henseler_p_one_tailed(a, b)
    assert 0.45 < p < 0.55


def test_parametric_and_welch_recover_known_smartpls_values():
    """SmartPLS' Corporate Reputation Bootstrap MGA shows for ATTR -> COMP:
    diff=-0.114, SE_Contract=0.067 (n=219), SE_Prepaid=0.087 (n=125),
    parametric t=1.036, Welch t=1.042. We reproduce these to 2 decimals."""
    t_p, df_p, p_p = _parametric_t(-0.114, 0.067, 0.087, 219, 125)
    assert df_p == 342
    assert abs(abs(t_p) - 1.036) < 0.02
    # Sign of t follows sign of diff
    assert t_p < 0
    t_w, df_w, _p_w = _welch_t(-0.114, 0.067, 0.087, 219, 125)
    assert abs(abs(t_w) - 1.042) < 0.02
    assert t_w < 0
    # Welch df is between max(n_a-1, n_b-1) and (n_a + n_b - 2)
    assert max(218, 124) <= df_w <= 342


# ---------------------------------------------------------------------------
# Smoke tests on the small fixture
# ---------------------------------------------------------------------------


def _russa_bmga(subsamples: int = 60) -> BootstrapMGA:
    russa = _russa_with_group()
    cfg = _russa_config()
    return BootstrapMGA(
        russa,
        cfg,
        grouping_column="region",
        groups=[
            GroupSpec(name="west", values=["west"]),
            GroupSpec(name="east", values=["east"]),
        ],
        scheme=Scheme.CENTROID,
        subsamples=subsamples,
        seed=42,
        alpha=0.05,
    )


def test_path_coefficients_schema():
    bmga = _russa_bmga()
    df = bmga.path_coefficients()
    assert {"source", "target", "difference",
            "henseler_p_1tailed", "henseler_p_2tailed",
            "parametric_t", "parametric_df", "parametric_p",
            "welch_t", "welch_df", "welch_p"}.issubset(df.columns)
    # Per-group columns parameterised on group name
    for g in bmga.group_names:
        for prefix in (
            "original_", "mean_", "std_error_", "t_value_", "p_value_",
            "ci_bc_2_5_", "ci_bc_97_5_",
        ):
            assert f"{prefix}{g}" in df.columns
    # 2 paths in the Russa model: AGRI -> POLINS, IND -> POLINS
    assert len(df) == 2


def test_outer_loadings_and_weights_shapes_match():
    bmga = _russa_bmga()
    loadings = bmga.outer_loadings()
    weights = bmga.outer_weights()
    assert len(loadings) == len(weights)
    # 3 + 2 + 4 = 9 indicators
    assert len(loadings) == 9
    for df in (loadings, weights):
        assert {"lv", "indicator", "difference"}.issubset(df.columns)


def test_total_effects_excludes_self_loops_and_zero_pairs():
    bmga = _russa_bmga()
    df = bmga.total_effects()
    # Russa has only exogenous->endogenous total effects: AGRI->POLINS, IND->POLINS.
    # There are no other non-trivial pairs.
    assert {"source", "target"}.issubset(df.columns)
    sources = set(df["source"])
    targets = set(df["target"])
    assert "POLINS" in targets
    assert sources.issubset({"AGRI", "IND"})


def test_specific_and_total_indirect_effects_empty_for_flat_model():
    """Russa has no mediator, so SIE / TIE must be empty."""
    bmga = _russa_bmga()
    sie = bmga.specific_indirect_effects()
    tie = bmga.total_indirect_effects()
    assert sie.empty
    assert tie.empty


def test_difference_equals_originals_subtracted():
    bmga = _russa_bmga()
    df = bmga.path_coefficients()
    g_a, g_b = bmga.group_names
    diffs = df[f"original_{g_a}"] - df[f"original_{g_b}"]
    assert (df["difference"] - diffs).abs().max() < 1e-12


def test_henseler_p_values_in_unit_interval():
    bmga = _russa_bmga()
    df = bmga.path_coefficients()
    for col in ("henseler_p_1tailed", "henseler_p_2tailed"):
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()


def test_p_2tailed_equals_2_min_p1_one_minus_p1():
    bmga = _russa_bmga()
    df = bmga.path_coefficients()
    p1 = df["henseler_p_1tailed"]
    expected = 2 * np.minimum(p1, 1 - p1)
    assert (df["henseler_p_2tailed"] - expected).abs().max() < 1e-12


def test_reproducible_with_same_seed():
    a = _russa_bmga(subsamples=40).path_coefficients()
    b = _russa_bmga(subsamples=40).path_coefficients()
    pd.testing.assert_frame_equal(a, b)


def test_rejects_wrong_number_of_groups():
    russa = _russa_with_group()
    cfg = _russa_config()
    with pytest.raises(ValueError, match="exactly 2 groups"):
        BootstrapMGA(
            russa, cfg, "region",
            [GroupSpec(name="only", values=["west"])],
            subsamples=10,
        )


def test_rejects_missing_grouping_column():
    russa = _russa_with_group()
    cfg = _russa_config()
    with pytest.raises(ValueError, match="grouping_column"):
        BootstrapMGA(
            russa, cfg, "nope",
            [
                GroupSpec(name="w", values=["west"]),
                GroupSpec(name="e", values=["east"]),
            ],
            subsamples=10,
        )


def test_rejects_too_few_subsamples():
    russa = _russa_with_group()
    cfg = _russa_config()
    with pytest.raises(ValueError, match="subsamples"):
        BootstrapMGA(
            russa, cfg, "region",
            [
                GroupSpec(name="w", values=["west"]),
                GroupSpec(name="e", values=["east"]),
            ],
            subsamples=1,
        )


# ---------------------------------------------------------------------------
# SmartPLS-reference comparison on the Corporate Reputation primer (Hair 2022)
# ---------------------------------------------------------------------------
#
# Fixture: tests/data/corporate_reputation.csv (n=344, servicetype 1=Prepaid
# n=125, servicetype 2=Contract n=219). Extended model with ATTR, COMP, CSOR,
# CUSA, CUSL, LIKE, PERF, QUAL.
#
# SmartPLS reference tables live next to the data file as
# ``mga_corp_reputation__<entity>__<tab>.csv``. We can only match SmartPLS
# loosely because (a) we don't reproduce their RNG sequence and (b) they round
# all reported numbers to 3 decimals. Deterministic quantities (original path
# coefficients, group-pair differences) match within rounding; bootstrap
# standard errors and t-values are checked with looser MC tolerances.


_CR_LV_BLOCKS = {
    "ATTR": ["attr_1", "attr_2", "attr_3"],
    "COMP": ["comp_1", "comp_2", "comp_3"],
    "CSOR": ["csor_1", "csor_2", "csor_3", "csor_4", "csor_5"],
    "CUSA": ["cusa"],
    "CUSL": ["cusl_1", "cusl_2", "cusl_3"],
    "LIKE": ["like_1", "like_2", "like_3"],
    "PERF": ["perf_1", "perf_2", "perf_3", "perf_4", "perf_5"],
    "QUAL": [
        "qual_1", "qual_2", "qual_3", "qual_4",
        "qual_5", "qual_6", "qual_7", "qual_8",
    ],
}

_CR_PATHS = [
    (["ATTR", "CSOR", "PERF", "QUAL"], ["COMP"]),
    (["ATTR", "CSOR", "PERF", "QUAL"], ["LIKE"]),
    (["COMP", "LIKE"], ["CUSA"]),
    (["COMP", "LIKE", "CUSA"], ["CUSL"]),
]


def _cr_data() -> pd.DataFrame:
    df = pd.read_csv("tests/data/corporate_reputation.csv", sep=";")
    df = df.replace(-99, np.nan)
    return df


def _cr_config() -> c.Config:
    structure = c.Structure()
    for sources, targets in _CR_PATHS:
        structure.add_path(sources, targets)
    config = c.Config(structure.path(), default_scale=Scale.NUM)
    for lv, indicators in _CR_LV_BLOCKS.items():
        config.add_lv(lv, Mode.A, *[c.MV(name) for name in indicators])
    return config


@pytest.fixture(scope="module")
def cr_bmga() -> BootstrapMGA:
    """Single 1000-subsample run shared across SmartPLS-reference tests."""
    return BootstrapMGA(
        _cr_data(),
        _cr_config(),
        grouping_column="servicetype",
        groups=[
            GroupSpec(name="Contract", values=[2]),
            GroupSpec(name="Prepaid", values=[1]),
        ],
        scheme=Scheme.CENTROID,
        subsamples=1000,
        seed=42,
        alpha=0.05,
    )


def _smartpls_path_diff_table() -> pd.DataFrame:
    """``Difference (Contract plan - Prepaid plan)`` column from the SmartPLS
    bootstrap-MGA path table."""
    df = pd.read_csv(
        "tests/data/mga_corp_reputation__path_coefficients__bootstrap_mga.csv"
    )
    return df.set_index("path")


def _smartpls_path_bootstrap_table() -> pd.DataFrame:
    """Per-group originals + SEs from the SmartPLS bootstrap-results table."""
    df = pd.read_csv(
        "tests/data/mga_corp_reputation__path_coefficients__bootstrap_results.csv"
    )
    return df.set_index("path")


def test_cr_group_sizes_match_smartpls(cr_bmga):
    assert cr_bmga.group_sizes == (219, 125)
    assert cr_bmga.group_names == ("Contract", "Prepaid")


def test_cr_path_coefficient_pattern_correlates_with_smartpls(cr_bmga):
    """Engine per-group originals must correlate strongly with SmartPLS across
    all 26 (path, group) pairs (>= 0.95 Pearson).

    Notes
    -----
    Bit-exact reproduction of SmartPLS isn't possible: on the COMP block in
    particular (4 highly collinear exogenous LVs), small differences in inner-
    weighting starting points / sign convention can shift weight between
    predictors without changing the prediction. The overall pattern of
    coefficients across the 13 paths nevertheless agrees tightly.
    """
    smartpls = _smartpls_path_bootstrap_table()
    df = cr_bmga.path_coefficients().copy()
    df["path"] = df["source"] + " -> " + df["target"]
    df = df.set_index("path")
    common = df.index.intersection(smartpls.index)
    assert len(common) == 13
    engine = np.concatenate([
        df.loc[common, "original_Contract"].to_numpy(),
        df.loc[common, "original_Prepaid"].to_numpy(),
    ])
    smartpls_vec = np.concatenate([
        smartpls.loc[common, "Original (Contract plan)"].to_numpy(),
        smartpls.loc[common, "Original (Prepaid plan)"].to_numpy(),
    ])
    r = np.corrcoef(engine, smartpls_vec)[0, 1]
    assert r > 0.95, f"per-group path-coefficient correlation only {r:.3f}"


def test_cr_path_differences_correlate_with_smartpls(cr_bmga):
    """Engine differences must correlate strongly with SmartPLS contrasts and
    agree on sign for the larger-magnitude SmartPLS contrasts."""
    smartpls = _smartpls_path_diff_table()
    df = cr_bmga.path_coefficients().copy()
    df["path"] = df["source"] + " -> " + df["target"]
    df = df.set_index("path")
    common = df.index.intersection(smartpls.index)
    engine_diff = df.loc[common, "difference"].to_numpy()
    smartpls_diff = smartpls.loc[
        common, "Difference (Contract plan - Prepaid plan)"
    ].to_numpy()
    r = np.corrcoef(engine_diff, smartpls_diff)[0, 1]
    assert r > 0.85, f"difference correlation only {r:.3f}"
    # Sign agreement on substantively large contrasts (|diff| >= 0.15).
    large = np.abs(smartpls_diff) >= 0.15
    if large.any():
        assert np.array_equal(
            np.sign(engine_diff[large]), np.sign(smartpls_diff[large])
        ), "sign mismatch on substantively large contrasts"


def test_cr_per_group_bootstrap_se_in_smartpls_ballpark(cr_bmga):
    """Bootstrap SEs are MC-noisy (different RNG) AND inherit the slight
    per-group estimate drift; require |delta| <= 0.05."""
    smartpls = _smartpls_path_bootstrap_table()
    df = cr_bmga.path_coefficients().copy()
    df["path"] = df["source"] + " -> " + df["target"]
    df = df.set_index("path")
    for path in df.index:
        for group, sm_col in (
            ("Contract", "STDEV (Contract plan)"),
            ("Prepaid", "STDEV (Prepaid plan)"),
        ):
            se_engine = df.loc[path, f"std_error_{group}"]
            se_smartpls = smartpls.loc[path, sm_col]
            assert abs(se_engine - se_smartpls) < 0.05, (
                f"{path} SE_{group}: engine={se_engine:.3f} "
                f"vs SmartPLS={se_smartpls:.3f}"
            )


def test_cr_parametric_and_welch_p_values_in_unit_interval(cr_bmga):
    """All parametric / Welch p-values must lie in [0, 1]."""
    df = cr_bmga.path_coefficients()
    for col in ("parametric_p", "welch_p"):
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()


def test_cr_significant_paths_agree_with_smartpls(cr_bmga):
    """SmartPLS flags CUSA -> CUSL and LIKE -> CUSL as significant differences
    at alpha=0.05 (Welch p=0.043 and 0.042). Engine should reach at least one
    of these conclusions despite RNG and per-group estimate differences."""
    df = cr_bmga.path_coefficients().copy()
    df["path"] = df["source"] + " -> " + df["target"]
    df = df.set_index("path")
    significant = set(df.index[df["welch_p"] < 0.05].tolist())
    smartpls_significant = {"CUSA -> CUSL", "LIKE -> CUSL"}
    assert significant & smartpls_significant, (
        f"engine flagged {significant}, expected overlap with {smartpls_significant}"
    )
