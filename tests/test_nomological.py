"""Nomological validity: directional hypothesis testing on LV correlations."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import stats

import openpls.config as c
from openpls.mode import Mode
from openpls.nomological import NomologicalValidity
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


def test_correlation_matches_pearson():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")])
    r_expected = float(fit.scores()["SAT"].corr(fit.scores()["LOY"]))
    r_actual = float(nv.table().loc["SAT -> LOY", "correlation"])
    npt.assert_allclose(r_actual, r_expected, atol=1e-12)


def test_t_statistic_matches_manual_formula():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")])
    row = nv.table().loc["SAT -> LOY"]
    n = len(fit.scores())
    r = float(row["correlation"])
    expected_t = r * np.sqrt(n - 2) / np.sqrt(1.0 - r * r)
    npt.assert_allclose(float(row["t_statistic"]), expected_t, atol=1e-10)
    assert int(row["df"]) == n - 2


def test_p_one_sided_matches_scipy():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")])
    row = nv.table().loc["SAT -> LOY"]
    n = len(fit.scores())
    expected_p = float(stats.t.sf(float(row["t_statistic"]), n - 2))
    npt.assert_allclose(float(row["p_one_sided"]), expected_p, atol=1e-12)


def test_negative_hypothesis_uses_lower_tail():
    """Sign '-' expects a negative correlation; p-value is left-tail."""
    scores = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    nv = NomologicalValidity(scores, [("A", "B", "-")])
    row = nv.table().loc["A -> B"]
    assert float(row["correlation"]) < 0
    # Perfect negative correlation → p should be tiny.
    assert float(row["p_one_sided"]) < 1e-6
    assert row["verdict"] == "supported"


def test_positive_hypothesis_on_positive_correlation_supported():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")], alpha=0.05)
    row = nv.table().loc["SAT -> LOY"]
    # SAT -> LOY is the strongest positive path in the ECSI model.
    assert float(row["correlation"]) > 0
    assert row["sign_matches"]
    assert row["verdict"] == "supported"


def test_sign_mismatch_gives_not_supported():
    fit = _fit_satisfaction()
    # Hypothesise a NEGATIVE correlation where the true correlation is
    # strongly positive. Verdict must be "not supported" regardless of
    # significance, because the sign does not match.
    nv = fit.nomological_validity([("SAT", "LOY", "-")])
    row = nv.table().loc["SAT -> LOY"]
    assert not row["sign_matches"]
    assert row["verdict"] == "not supported"


def test_zero_correlation_not_supported():
    scores = pd.DataFrame({
        "A": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        "B": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    })
    nv = NomologicalValidity(scores, [("A", "B", "+")])
    row = nv.table().loc["A -> B"]
    npt.assert_allclose(float(row["correlation"]), 0.0, atol=1e-12)
    assert row["verdict"] == "not supported"


def test_table_index_is_arrow_labels():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([
        ("IMAG", "SAT", "+"),
        ("SAT", "LOY", "+"),
        ("QUAL", "VAL", "+"),
    ])
    assert list(nv.table().index) == ["IMAG -> SAT", "SAT -> LOY", "QUAL -> VAL"]


def test_correlations_matrix_is_symmetric_with_unit_diagonal():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")])
    corr = nv.correlations()
    assert list(corr.index) == list(corr.columns)
    npt.assert_allclose(np.diag(corr.values), 1.0, atol=1e-12)
    npt.assert_allclose(corr.values, corr.values.T, atol=1e-12)


def test_alpha_stored_and_respected():
    fit = _fit_satisfaction()
    nv = fit.nomological_validity([("SAT", "LOY", "+")], alpha=0.01)
    assert nv.alpha() == 0.01


def test_alpha_boundary_rejects_supported_when_p_above_threshold():
    """A weak positive correlation should be 'supported' at α=0.10 but not α=0.001."""
    rng = np.random.default_rng(seed=7)
    n = 40
    a = rng.standard_normal(n)
    # Weak positive: mostly noise plus a whisper of a.
    b = 0.25 * a + rng.standard_normal(n)
    scores = pd.DataFrame({"A": a, "B": b})

    lenient = NomologicalValidity(scores, [("A", "B", "+")], alpha=0.10)
    strict = NomologicalValidity(scores, [("A", "B", "+")], alpha=0.001)

    r_lenient = float(lenient.table().loc["A -> B", "correlation"])
    r_strict = float(strict.table().loc["A -> B", "correlation"])
    assert r_lenient > 0
    npt.assert_allclose(r_lenient, r_strict, atol=1e-12)

    # Same p, different α threshold → different verdicts possible.
    p = float(lenient.table().loc["A -> B", "p_one_sided"])
    assert lenient.table().loc["A -> B", "verdict"] == (
        "supported" if p < 0.10 else "not supported"
    )
    assert strict.table().loc["A -> B", "verdict"] == (
        "supported" if p < 0.001 else "not supported"
    )


def test_invalid_sign_raises():
    fit = _fit_satisfaction()
    with pytest.raises(ValueError, match="expected sign"):
        fit.nomological_validity([("SAT", "LOY", "?")])


def test_source_equals_target_raises():
    fit = _fit_satisfaction()
    with pytest.raises(ValueError, match="source and target"):
        fit.nomological_validity([("SAT", "SAT", "+")])


def test_unknown_lv_raises():
    fit = _fit_satisfaction()
    with pytest.raises(KeyError):
        fit.nomological_validity([("SAT", "MISSING", "+")])


def test_alpha_out_of_range_raises():
    fit = _fit_satisfaction()
    with pytest.raises(ValueError, match="alpha"):
        fit.nomological_validity([("SAT", "LOY", "+")], alpha=1.5)
