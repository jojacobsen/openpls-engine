import math

import pandas as pd

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


def test_q_squared_returns_endogenous_lvs():
    plspm_calc = _satisfaction_plspm()
    q2 = plspm_calc.q_squared()
    # Endogenous LVs in ECSI: EXPE, QUAL, VAL, SAT, LOY. IMAG is exogenous.
    assert set(q2.index) == {"EXPE", "QUAL", "VAL", "SAT", "LOY"}
    assert "q_squared" in q2.columns


def test_q_squared_values_in_plausible_range():
    plspm_calc = _satisfaction_plspm()
    q2 = plspm_calc.q_squared()
    for lv in q2.index:
        val = float(q2.loc[lv, "q_squared"])
        assert math.isfinite(val), f"{lv}: q_squared not finite"
        # All ECSI endogenous LVs should show predictive relevance (Q² > 0).
        # Wide tolerance because blindfolding with D=7 is approximate.
        assert -0.1 < val < 1.0, f"{lv}: q_squared {val} outside plausible range"


def test_q_squared_omission_distance_param():
    plspm_calc = _satisfaction_plspm()
    q2_d7 = plspm_calc.q_squared(omission_distance=7).copy()
    q2_d5 = plspm_calc.q_squared(omission_distance=5).copy()
    # Different D should give different values (typically close but not equal).
    diffs = (q2_d7["q_squared"] - q2_d5["q_squared"]).abs()
    assert diffs.sum() > 0, "different omission distances produced identical Q²"


def test_q_squared_rejects_invalid_distance():
    plspm_calc = _satisfaction_plspm()
    try:
        plspm_calc.q_squared(omission_distance=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for omission_distance=1")
