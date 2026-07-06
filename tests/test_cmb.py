"""Common Method Bias — Lindell & Whitney (2001) marker-variable diagnostic."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import stats

import openpls.config as c
from openpls.cmb import CMBLindellWhitney
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
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return Plspm(satisfaction, config)


def _independent_marker(scores: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(len(scores)), index=scores.index, name="marker")


def test_marker_correlations_match_pearson():
    fit = _fit_satisfaction()
    scores = fit.scores()
    marker = _independent_marker(scores)
    result = fit.cmb_lindell_whitney(marker)
    for lv in scores.columns:
        expected = float(scores[lv].corr(marker))
        actual = float(result.marker_correlations().loc[lv])
        npt.assert_allclose(actual, expected, atol=1e-12)


def test_cmb_estimate_is_smallest_absolute_marker_correlation():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    correlations = result.marker_correlations().dropna()
    idx = correlations.abs().idxmin()
    npt.assert_allclose(result.cmb_estimate(), float(correlations.loc[idx]), atol=1e-12)


def test_unadjusted_matches_scores_pearson_correlation():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    expected = fit.scores().corr(method="pearson")
    pd.testing.assert_frame_equal(result.unadjusted(), expected)


def test_adjustment_formula():
    """Off-diagonal r_A = (r_U - r_M) / (1 - r_M)."""
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    r_m = result.cmb_estimate()
    unadjusted = result.unadjusted()
    adjusted = result.adjusted()
    lvs = list(unadjusted.columns)
    for i, lv_a in enumerate(lvs):
        for lv_b in lvs[i + 1 :]:
            r_u = float(unadjusted.loc[lv_a, lv_b])
            expected = (r_u - r_m) / (1.0 - r_m)
            actual = float(adjusted.loc[lv_a, lv_b])
            npt.assert_allclose(actual, expected, atol=1e-12)


def test_adjusted_diagonal_is_one():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    for lv in result.adjusted().index:
        npt.assert_allclose(float(result.adjusted().loc[lv, lv]), 1.0, atol=1e-12)


def test_table_p_values_match_scipy():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    n = len(fit.scores())
    for _, row in result.table().iterrows():
        r_u = float(row["r_unadjusted"])
        r_a = float(row["r_adjusted"])
        t_u = r_u * np.sqrt(n - 2) / np.sqrt(1.0 - r_u * r_u)
        t_a = r_a * np.sqrt(n - 3) / np.sqrt(1.0 - r_a * r_a)
        p_u_expected = float(2.0 * stats.t.sf(abs(t_u), n - 2))
        p_a_expected = float(2.0 * stats.t.sf(abs(t_a), n - 3))
        npt.assert_allclose(float(row["p_unadjusted"]), p_u_expected, atol=1e-10)
        npt.assert_allclose(float(row["p_adjusted"]), p_a_expected, atol=1e-10)


def test_table_pair_index_and_columns():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker)
    table = result.table()
    lvs = list(fit.scores().columns)
    expected_pairs = len(lvs) * (len(lvs) - 1) // 2
    assert len(table) == expected_pairs
    for idx in table.index:
        assert " <-> " in idx
    expected_cols = {
        "lv_a",
        "lv_b",
        "r_unadjusted",
        "p_unadjusted",
        "sig_unadjusted",
        "r_adjusted",
        "p_adjusted",
        "sig_adjusted",
        "verdict",
    }
    assert expected_cols.issubset(set(table.columns))


def test_verdict_lost_significance_when_marker_shrinks_weak_pair():
    """A borderline-significant pair loses significance after a shared-method adjustment."""
    n = 500
    rng = np.random.default_rng(6)
    common = rng.standard_normal(n)
    a = 0.30 * common + rng.standard_normal(n)
    b = 0.30 * common + rng.standard_normal(n)
    c = 0.30 * common + rng.standard_normal(n)
    scores = pd.DataFrame({"A": a, "B": b, "C": c})
    marker = pd.Series(0.30 * common + rng.standard_normal(n), name="marker")
    result = CMBLindellWhitney(scores, marker)
    ab = result.table().loc["A <-> B"]
    assert ab["sig_unadjusted"]
    assert not ab["sig_adjusted"]
    assert ab["verdict"] == "lost_significance"


def test_verdict_unchanged_when_marker_uncorrelated():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores(), seed=123)
    result = fit.cmb_lindell_whitney(marker)
    # r_M is tiny → adjusted correlations barely move → significant pairs stay significant.
    assert abs(result.cmb_estimate()) < 0.15
    lost = (result.table()["verdict"] == "lost_significance").sum()
    assert lost == 0


def test_pairwise_deletion_on_marker_nan():
    scores = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0],
    })
    marker = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    result = CMBLindellWhitney(scores, marker)
    corrs = result.marker_correlations()
    expected_a = pd.concat([scores["A"], marker], axis=1).dropna().corr().iloc[0, 1]
    npt.assert_allclose(float(corrs.loc["A"]), float(expected_a), atol=1e-12)


def test_alpha_is_stored():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    result = fit.cmb_lindell_whitney(marker, alpha=0.01)
    assert result.alpha() == 0.01


def test_alpha_boundary_rejected():
    fit = _fit_satisfaction()
    marker = _independent_marker(fit.scores())
    with pytest.raises(ValueError, match="alpha"):
        fit.cmb_lindell_whitney(marker, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        fit.cmb_lindell_whitney(marker, alpha=1.0)


def test_marker_too_few_observations_raises():
    scores = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    marker = pd.Series([np.nan, np.nan, np.nan, 1.0, 2.0])
    with pytest.raises(ValueError, match="three"):
        CMBLindellWhitney(scores, marker)


def test_marker_reindex_to_scores_index():
    """Marker with extra rows should be reindexed to scores, not error."""
    scores = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0]},
        index=[10, 20, 30, 40, 50],
    )
    marker = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[10, 20, 30, 40, 50])
    result = CMBLindellWhitney(scores, marker)
    assert np.isfinite(result.cmb_estimate())
