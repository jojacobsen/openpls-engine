import math

import numpy as np
import pandas as pd

import plspm.config as c
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scheme import Scheme


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
    return Plspm(satisfaction, config, Scheme.CENTROID), satisfaction


def test_inner_summary_has_adj_r_and_bic_columns():
    plspm_calc, _ = _satisfaction_plspm()
    summary = plspm_calc.inner_summary()
    for col in ("r_squared", "r_squared_adj", "bic", "ave"):
        assert col in summary.columns, f"missing column: {col}"


def test_adjusted_r_squared_matches_formula():
    plspm_calc, satisfaction = _satisfaction_plspm()
    summary = plspm_calc.inner_summary()
    n = len(satisfaction)
    # Path matrix order: IMAG (0 predictors), EXPE (1), QUAL (1), VAL (2),
    # SAT (4), LOY (2).
    predictors = {"IMAG": 0, "EXPE": 1, "QUAL": 1, "VAL": 2, "SAT": 4, "LOY": 2}
    for lv, k in predictors.items():
        r2 = float(summary.loc[lv, "r_squared"])
        adj = float(summary.loc[lv, "r_squared_adj"])
        if k == 0:
            assert adj == 0.0  # exogenous LV — formula not applied
        else:
            expected = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
            assert math.isclose(adj, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_bic_only_set_for_endogenous_lvs():
    plspm_calc, satisfaction = _satisfaction_plspm()
    summary = plspm_calc.inner_summary()
    # IMAG is exogenous: BIC undefined.
    assert pd.isna(summary.loc["IMAG", "bic"])
    # Endogenous LVs: BIC finite.
    for lv in ("EXPE", "QUAL", "VAL", "SAT", "LOY"):
        bic = summary.loc[lv, "bic"]
        assert not pd.isna(bic), f"{lv}: BIC unexpectedly NaN"
        assert math.isfinite(float(bic))


def test_bic_value_matches_formula():
    plspm_calc, satisfaction = _satisfaction_plspm()
    summary = plspm_calc.inner_summary()
    n = len(satisfaction)
    predictors = {"EXPE": 1, "QUAL": 1, "VAL": 2, "SAT": 4, "LOY": 2}
    for lv, k in predictors.items():
        r2 = float(summary.loc[lv, "r_squared"])
        sse = max((1.0 - r2) * (n - 1), 1e-12)
        expected = n * np.log(sse / n) + (k + 1) * np.log(n)
        assert math.isclose(float(summary.loc[lv, "bic"]), expected, rel_tol=1e-9, abs_tol=1e-9)
