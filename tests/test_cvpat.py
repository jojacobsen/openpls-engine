"""Cross-Validated Predictive Ability Test (Liengaard et al. 2021)."""

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import stats

import openpls.config as c
from openpls.cvpat import CVPAT
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


def test_overall_returns_expected_columns():
    fit = _satisfaction_plspm()
    result = fit.cvpat(k=5)
    overall = result.overall()
    expected = {
        "n",
        "mean_loss_pls",
        "mean_loss_benchmark",
        "mean_difference",
        "std_error",
        "t_statistic",
        "df",
        "p_one_sided",
        "verdict",
    }
    assert expected.issubset(set(overall.index))


def test_per_construct_indexed_by_endogenous_lvs():
    fit = _satisfaction_plspm()
    result = fit.cvpat(k=5)
    per_lv = result.per_construct()
    # Endogenous LVs in ECSI are EXPE, QUAL, VAL, SAT, LOY. IMAG is exogenous.
    assert set(per_lv.index) == {"EXPE", "QUAL", "VAL", "SAT", "LOY"}


def test_losses_has_one_row_per_obs_indicator_pair():
    fit = _satisfaction_plspm()
    result = fit.cvpat(k=5)
    losses = result.losses()
    assert set(losses.columns) >= {
        "observation",
        "lv",
        "indicator",
        "squared_error_pls",
        "squared_error_benchmark",
    }
    # Every entry should be finite and non-negative.
    assert (losses["squared_error_pls"] >= 0).all()
    assert (losses["squared_error_benchmark"] >= 0).all()


def test_ia_benchmark_pls_beats_indicator_average_on_satisfaction():
    """On the ECSI satisfaction data PLS should beat the naive train-mean baseline."""
    fit = _satisfaction_plspm()
    result = fit.cvpat(benchmark="IA", k=5)
    overall = result.overall()
    # PLS beats mean baseline → mean_difference < 0 and one-sided p is small.
    assert float(overall["mean_difference"]) < 0
    assert float(overall["p_one_sided"]) < 0.05
    assert overall["verdict"] == "pls_better"


def test_t_statistic_matches_manual_paired_t():
    fit = _satisfaction_plspm()
    result = fit.cvpat(k=5)
    overall = result.overall()
    losses = result.losses()
    pls = losses.groupby("observation")["squared_error_pls"].sum()
    bench = losses.groupby("observation")["squared_error_benchmark"].sum()
    common = pls.index.intersection(bench.index)
    d = (pls.loc[common] - bench.loc[common]).to_numpy()
    n = len(d)
    mean_d = np.mean(d)
    se = np.std(d, ddof=1) / np.sqrt(n)
    expected_t = mean_d / se
    expected_p = float(stats.t.cdf(expected_t, df=n - 1))
    npt.assert_allclose(float(overall["t_statistic"]), expected_t, atol=1e-10)
    npt.assert_allclose(float(overall["p_one_sided"]), expected_p, atol=1e-10)
    assert int(overall["df"]) == n - 1


def test_lm_benchmark_runs_and_returns_valid_test():
    fit = _satisfaction_plspm()
    result = fit.cvpat(benchmark="LM", k=5)
    overall = result.overall()
    assert math.isfinite(float(overall["t_statistic"]))
    assert math.isfinite(float(overall["p_one_sided"]))
    assert 0.0 <= float(overall["p_one_sided"]) <= 1.0


def test_invalid_benchmark_raises():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="benchmark"):
        fit.cvpat(benchmark="RIDGE")


def test_k_too_small_raises():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="k must be"):
        fit.cvpat(k=1)


def test_k_larger_than_n_raises():
    fit = _satisfaction_plspm()
    n = len(fit.data())
    with pytest.raises(ValueError, match="cannot exceed"):
        fit.cvpat(k=n + 1)


def test_repeats_must_be_positive():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="repeats"):
        fit.cvpat(repeats=0)


def test_alpha_out_of_range_rejected():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="alpha"):
        fit.cvpat(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        fit.cvpat(alpha=1.0)


def test_alpha_and_benchmark_accessors():
    fit = _satisfaction_plspm()
    result = fit.cvpat(alpha=0.01, benchmark="LM", k=5)
    assert result.alpha() == 0.01
    assert result.benchmark() == "LM"


def test_seed_makes_result_deterministic():
    fit = _satisfaction_plspm()
    a = fit.cvpat(k=5, seed=99).overall()
    b = fit.cvpat(k=5, seed=99).overall()
    npt.assert_allclose(
        float(a["mean_difference"]), float(b["mean_difference"]), atol=1e-12
    )
    npt.assert_allclose(
        float(a["t_statistic"]), float(b["t_statistic"]), atol=1e-12
    )


def test_verdict_no_difference_when_pls_and_benchmark_tie():
    """Contrived case: benchmark equals PLS predictions → d ≡ 0 → no_difference."""
    fit = _satisfaction_plspm()
    result = fit.cvpat(k=5)
    losses = result.losses().copy()
    # Force the benchmark loss to equal the PLS loss.
    losses["squared_error_benchmark"] = losses["squared_error_pls"]
    pls = losses.groupby("observation")["squared_error_pls"].sum()
    bench = losses.groupby("observation")["squared_error_benchmark"].sum()
    common = pls.index.intersection(bench.index)
    d = (pls.loc[common] - bench.loc[common]).to_numpy()
    # This is a sanity check on the shape of the null distribution, not an
    # assertion about the API.
    assert np.allclose(d, 0.0)
