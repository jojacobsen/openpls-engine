import math

import numpy as np
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


def test_htmt_matrix_is_symmetric_with_nan_diagonal():
    htmt = _satisfaction_plspm().htmt().matrix()
    assert list(htmt.index) == list(htmt.columns)
    assert htmt.shape == (6, 6)
    # Diagonal undefined.
    for lv in htmt.index:
        assert pd.isna(htmt.loc[lv, lv])
    # Symmetric off-diagonal.
    arr = htmt.to_numpy()
    upper = arr[np.triu_indices(6, k=1)]
    lower = arr.T[np.triu_indices(6, k=1)]
    np.testing.assert_allclose(upper, lower, equal_nan=True)


def test_htmt_values_in_plausible_range():
    htmt = _satisfaction_plspm().htmt().matrix()
    off_diag = htmt.to_numpy()[np.triu_indices(6, k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    # HTMT is a ratio of mean absolute correlations; for a well-defined
    # reflective model values are positive and rarely exceed ~1.1.
    assert (off_diag > 0).all()
    assert (off_diag < 1.5).all()
    # All 15 unique LV pairs should resolve since every LV has ≥3 indicators.
    assert len(off_diag) == 15


def test_htmt_pairs_long_format():
    htmt = _satisfaction_plspm().htmt().pairs()
    assert list(htmt.columns) == ["lv_a", "lv_b", "htmt"]
    # 6 LVs → C(6,2) = 15 pairs.
    assert len(htmt) == 15
    # All HTMT values finite and positive.
    assert htmt["htmt"].notna().all()
    assert (htmt["htmt"] > 0).all()


def test_htmt_skips_single_indicator_lv():
    """A single-indicator LV cannot have a within-block mean, so any pair
    involving it should be NaN in the HTMT matrix."""
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)

    # Build a tiny synthetic model with one single-indicator LV.
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    # EXPE has 5 indicators expe1..expe5; replace by a single-indicator manual config.
    from openpls.config import MV
    config.add_lv("EXPE", Mode.A, MV("expe1"))

    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    htmt = plspm_calc.htmt().matrix()
    assert math.isnan(htmt.loc["IMAG", "EXPE"])
    assert math.isnan(htmt.loc["EXPE", "IMAG"])
    # The pairs() view should drop the NaN pair.
    assert plspm_calc.htmt().pairs().empty


def test_htmt_skips_formative_lvs():
    """HTMT is only defined for reflectively measured (Mode A) constructs.
    Pairs that involve a Mode B (formative) LV must be NaN.
    """
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    # IMAG declared as formative; the other five remain reflective.
    config.add_lv_with_columns_named("IMAG", Mode.B, satisfaction, "imag")
    for lv in ["EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())

    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)
    htmt = plspm_calc.htmt().matrix()
    # Every pair involving IMAG is NaN.
    for other in ["EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        assert math.isnan(htmt.loc["IMAG", other])
        assert math.isnan(htmt.loc[other, "IMAG"])
    # The remaining 5 reflective LVs still produce 10 finite pairs.
    pairs = plspm_calc.htmt().pairs()
    assert len(pairs) == 10
    names = set(pairs["lv_a"]).union(set(pairs["lv_b"]))
    assert "IMAG" not in names
