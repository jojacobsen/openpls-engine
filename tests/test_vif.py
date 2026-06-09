import math

import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.config import MV
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme
from openpls.vif import _vif_one


def _satisfaction_plspm(mode: Mode = Mode.A) -> Plspm:
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, mode, satisfaction, lv.lower())
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_vif_items_table_shape_and_columns():
    items = _satisfaction_plspm().vif().items()
    assert list(items.columns) == ["lv", "indicator", "vif"]
    # Every multi-indicator block contributes one row per indicator.
    # IMAG (5) + EXPE (5) + QUAL (5) + VAL (4) + SAT (4) + LOY (4) = 27
    assert len(items) == 27
    assert items["vif"].notna().all()
    # Standard satisfaction data is highly correlated within blocks but
    # VIFs should be positive and finite.
    assert (items["vif"] > 0).all()
    assert np.isfinite(items["vif"]).all()


def test_vif_items_skips_single_indicator_blocks():
    """Single-indicator blocks have no within-block VIF; they're omitted."""
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv("EXPE", Mode.A, MV("expe1"))
    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)

    items = plspm_calc.vif().items()
    # IMAG has 5 indicators → 5 rows. EXPE has 1 → dropped.
    assert set(items["lv"].unique()) == {"IMAG"}
    assert len(items) == 5


def test_vif_inner_endogenous_only_with_multiple_predictors():
    """SAT has 4 predictors (IMAG, EXPE, QUAL, VAL); VAL has 2 (EXPE, QUAL);
    LOY has 2 (IMAG, SAT); EXPE has 1 (IMAG, dropped); QUAL has 1 (EXPE,
    dropped). So inner() reports for SAT, VAL, LOY."""
    inner = _satisfaction_plspm().vif().inner()
    assert set(inner.keys()) == {"SAT", "VAL", "LOY"}
    for endo, table in inner.items():
        assert list(table.columns) == ["predictor", "vif"]
        assert table["vif"].notna().all()
        assert (table["vif"] >= 1.0 - 1e-9).all()


def test_vif_inner_two_predictor_case_matches_one_over_one_minus_r_sq():
    """For exactly two predictors VIF = 1 / (1 - r²) where r is their
    correlation. Verified on VAL ← (EXPE, QUAL)."""
    plspm_calc = _satisfaction_plspm()
    scores = plspm_calc.scores()
    r = float(scores[["EXPE", "QUAL"]].corr().iloc[0, 1])
    expected = 1.0 / (1.0 - r ** 2)
    inner = plspm_calc.vif().inner()["VAL"]
    for _, row in inner.iterrows():
        np.testing.assert_allclose(row["vif"], expected, rtol=1e-9)


def test_vif_one_perfect_collinearity_returns_inf():
    """A regressor identical to (a linear transform of) the response gives
    R² = 1 and VIF = inf."""
    n = 50
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    x = np.column_stack([y, rng.normal(size=n)])
    assert math.isinf(_vif_one(y, x))


def test_vif_one_zero_variance_response_returns_nan():
    n = 20
    y = np.full(n, 3.0)
    x = np.random.default_rng(0).normal(size=(n, 2))
    assert math.isnan(_vif_one(y, x))


def test_vif_is_cached_across_calls():
    plspm_calc = _satisfaction_plspm()
    a = plspm_calc.vif()
    b = plspm_calc.vif()
    assert a is b


def test_vif_works_with_mode_b_formative_block():
    """Mode B is the canonical use case for VIF — make sure the API
    doesn't choke on a formative model."""
    items = _satisfaction_plspm(mode=Mode.B).vif().items()
    assert len(items) == 27
    assert items["vif"].notna().all()
    assert (items["vif"] > 0).all()
