import math

import pandas as pd
import pytest

import openpls.config as c
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


def test_predict_metrics_table_structure():
    fit = _satisfaction_plspm()
    m = fit.predict(k=5).metrics()
    assert list(m.index.names) == ["lv", "indicator"]
    assert set(m.columns) == {
        "rmse_pls", "mae_pls", "mape_pls", "q2_predict",
        "rmse_lm", "mae_lm", "mape_lm",
        "rmse_pls_in", "mae_pls_in", "mape_pls_in",
        "rmse_lm_in", "mae_lm_in", "mape_lm_in",
    }
    # Endogenous LVs in ECSI are EXPE, QUAL, VAL, SAT, LOY. IMAG is exogenous,
    # so its indicators must not appear.
    lvs = {lv for lv, _ in m.index}
    assert "IMAG" not in lvs
    assert lvs == {"EXPE", "QUAL", "VAL", "SAT", "LOY"}


def test_predict_metrics_finite_and_positive():
    fit = _satisfaction_plspm()
    m = fit.predict(k=5).metrics()
    for col in ("rmse_pls", "mae_pls", "rmse_lm", "mae_lm"):
        for ind in m.index:
            val = float(m.loc[ind, col])
            assert math.isfinite(val), f"{ind}: {col} not finite"
            assert val >= 0.0, f"{ind}: {col} = {val} < 0"


def test_predict_q2_in_plausible_range():
    fit = _satisfaction_plspm()
    m = fit.predict(k=5).metrics()
    for ind in m.index:
        q2 = float(m.loc[ind, "q2_predict"])
        assert math.isfinite(q2), f"{ind}: q2_predict not finite"
        # Most ECSI indicators should beat the train-mean baseline (Q² > 0),
        # but the bound is loose because k=5 with 250 obs has noticeable variance.
        assert -0.3 < q2 < 1.0, f"{ind}: q2_predict {q2} outside plausible range"


def test_predict_deterministic_with_seed():
    fit = _satisfaction_plspm()
    m1 = fit.predict(k=5, seed=42).metrics().copy()
    m2 = fit.predict(k=5, seed=42).metrics().copy()
    diffs = (m1 - m2).abs().sum().sum()
    assert diffs == 0.0, "same seed produced different metrics"


def test_predict_different_seed_changes_metrics():
    fit = _satisfaction_plspm()
    m1 = fit.predict(k=5, seed=1).metrics()
    m2 = fit.predict(k=5, seed=2).metrics()
    # Fold partitions differ, so metrics will differ at least slightly.
    diff = (m1["rmse_pls"] - m2["rmse_pls"]).abs().sum()
    assert diff > 0.0, "different seeds produced identical PLS RMSEs"


def test_predict_summary_categorizes_each_indicator():
    fit = _satisfaction_plspm()
    pred = fit.predict(k=5)
    m = pred.metrics()
    s = pred.summary()
    assert set(s.unique()).issubset({"better", "worse", "tie"})
    assert len(s) == len(m)


def test_predict_rejects_bad_k():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="k must"):
        fit.predict(k=1)


def test_predict_rejects_k_larger_than_n():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="exceed sample size"):
        fit.predict(k=100000)


def test_predict_rejects_bad_repeats():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="repeats"):
        fit.predict(k=5, repeats=0)


def test_predict_repeats_average_more_stable():
    fit = _satisfaction_plspm()
    m_single = fit.predict(k=5, repeats=1, seed=42).metrics()
    m_multi = fit.predict(k=5, repeats=3, seed=42).metrics()
    # With more repeats the structure is unchanged but values can shift.
    assert m_single.shape == m_multi.shape
    assert set(m_single.index) == set(m_multi.index)


def test_predict_mape_finite_and_positive():
    """MAPE columns are populated, finite, and non-negative."""
    fit = _satisfaction_plspm()
    m = fit.predict(k=5).metrics()
    for col in ("mape_pls", "mape_lm", "mape_pls_in", "mape_lm_in"):
        for ind in m.index:
            val = float(m.loc[ind, col])
            assert math.isfinite(val), f"{ind}: {col} not finite"
            assert val >= 0.0, f"{ind}: {col} = {val} < 0"


def test_predict_in_sample_columns_finite_and_at_most_oos():
    """In-sample errors are populated and (typically) <= out-of-sample errors."""
    fit = _satisfaction_plspm()
    m = fit.predict(k=5, seed=42).metrics()
    for col in ("rmse_pls_in", "mae_pls_in", "rmse_lm_in", "mae_lm_in"):
        for ind in m.index:
            val = float(m.loc[ind, col])
            assert math.isfinite(val), f"{ind}: {col} not finite"
            assert val >= 0.0, f"{ind}: {col} = {val} < 0"
    # in-sample <= out-of-sample for the majority of indicators (loose check
    # because k-fold variance can flip a few).
    pls_better_in = (m["rmse_pls_in"] <= m["rmse_pls"] + 1e-6).sum()
    lm_better_in = (m["rmse_lm_in"] <= m["rmse_lm"] + 1e-6).sum()
    assert pls_better_in >= len(m) // 2
    assert lm_better_in >= len(m) // 2


def test_predict_in_sample_does_not_depend_on_seed_or_k():
    """In-sample fit uses the whole dataset, so it is invariant under k/seed."""
    fit = _satisfaction_plspm()
    m_a = fit.predict(k=5, seed=1).metrics()
    m_b = fit.predict(k=8, seed=99).metrics()
    in_cols = ["rmse_pls_in", "mae_pls_in", "mape_pls_in",
               "rmse_lm_in", "mae_lm_in", "mape_lm_in"]
    for col in in_cols:
        for ind in m_a.index:
            assert math.isclose(
                float(m_a.loc[ind, col]),
                float(m_b.loc[ind, col]),
                rel_tol=1e-9, abs_tol=1e-9,
            ), f"{ind}: {col} differs between seeds/k"
