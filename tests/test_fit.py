import math

import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def satisfaction_path_matrix():
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    return structure.path()


def _satisfaction_plspm():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_model_fit_srmr_in_reasonable_range():
    plspm_calc = _satisfaction_plspm()
    fit = plspm_calc.model_fit()
    srmr = fit.srmr()
    assert math.isfinite(srmr)
    # ECSI / satisfaction is the canonical mode-A example. SRMR should be
    # comfortably below the conventional 0.10 acceptable-fit cutoff.
    assert 0.0 <= srmr <= 0.10, f"SRMR={srmr} outside expected range"


def test_model_fit_duls_nonnegative():
    fit = _satisfaction_plspm().model_fit()
    d_uls = fit.d_uls()
    assert math.isfinite(d_uls)
    assert d_uls >= 0.0


def test_model_fit_residuals_shape():
    fit = _satisfaction_plspm().model_fit()
    resid = fit.residuals()
    # 24 indicators total: 5 for each of IMAG, EXPE, QUAL + 3 for VAL + 3 for SAT + 4 for LOY
    assert resid.shape[0] == resid.shape[1]
    assert resid.shape[0] >= 20
    # Residual matrix is symmetric.
    pd_diff = (resid - resid.T).abs().to_numpy().max()
    assert pd_diff < 1e-10


def test_model_fit_summary_has_both_metrics():
    summary = _satisfaction_plspm().model_fit().summary()
    assert list(summary.columns) == ["srmr", "d_uls"]
    assert summary.shape == (1, 2)
