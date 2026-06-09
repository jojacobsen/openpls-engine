import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.config import MV
from openpls.cta import CTAPLS, _canonical_tetrad
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


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


def _one_factor_data(
    n: int = 400,
    p: int = 5,
    loadings: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate ``p`` indicators of a single common factor.

    ``x_j = lambda_j * f + e_j`` with ``f ~ N(0, 1)`` and
    ``e_j ~ N(0, 1 - lambda_j^2)``. Implies all tetrads vanish in
    expectation, so CTA-PLS should fail to reject the reflective
    specification on this data.
    """
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(n)
    err_var = max(1.0 - loadings ** 2, 1e-6)
    cols = {
        f"x{j+1}": loadings * f + rng.normal(scale=np.sqrt(err_var), size=n)
        for j in range(p)
    }
    return pd.DataFrame(cols)


def _independent_data(n: int = 400, p: int = 5, seed: int = 0) -> pd.DataFrame:
    """Independent Gaussians: every covariance is ~0, so all tetrads are also
    near zero. CTA-PLS should not reject — this is a sanity bound on the
    false-positive rate, not a counter-example."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"x{j+1}": rng.standard_normal(n) for j in range(p)})


def _formative_like_data(
    n: int = 400, p: int = 5, seed: int = 1
) -> pd.DataFrame:
    """Construct indicators whose covariance structure is not a one-factor
    model: indicators load on two roughly orthogonal factors. Tetrads should
    not vanish — CTA-PLS should reject for at least one tetrad."""
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    # Half the indicators load on f1 only, half on f2 only.
    half = p // 2
    cols = {}
    for j in range(p):
        load_f1 = 0.85 if j < half else 0.0
        load_f2 = 0.0 if j < half else 0.85
        signal = load_f1 * f1 + load_f2 * f2
        noise = rng.normal(scale=0.5, size=n)
        cols[f"x{j+1}"] = signal + noise
    return pd.DataFrame(cols)


def _cta_on_single_block(data: pd.DataFrame, **kwargs) -> CTAPLS:
    """Wrap a single block of simulated indicators in a minimal Config so we
    can exercise CTAPLS without spinning up Plspm. Adds an extra column to
    act as a dummy second LV (any non-zero, distinct variable will do)."""
    full = data.copy()
    rng = np.random.default_rng(123)
    full["dv1"] = rng.standard_normal(len(full))
    structure = c.Structure()
    structure.add_path(["IV"], ["DV"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv("IV", Mode.A, *[MV(col) for col in data.columns])
    config.add_lv("DV", Mode.A, MV("dv1"))
    return CTAPLS(config, full, **kwargs)


def test_canonical_tetrad_matches_definition():
    cov = np.array(
        [
            [1.0, 0.5, 0.4, 0.3],
            [0.5, 1.0, 0.6, 0.2],
            [0.4, 0.6, 1.0, 0.7],
            [0.3, 0.2, 0.7, 1.0],
        ]
    )
    # s_01 * s_23 - s_02 * s_13 = 0.5*0.7 - 0.4*0.2 = 0.27
    assert _canonical_tetrad(cov, 0, 1, 2, 3) == pytest.approx(0.27)


def test_cta_table_shape_and_columns():
    cta = _satisfaction_plspm().cta(n_boot=100, seed=0)
    tetrads = cta.tetrads()
    summary = cta.summary()
    assert list(tetrads.columns) == [
        "lv",
        "indicators",
        "tetrad",
        "boot_se",
        "p_value",
        "holm_decision",
    ]
    assert list(summary.columns) == [
        "lv",
        "n_indicators",
        "n_tetrads",
        "n_rejected",
        "decision",
    ]
    # IMAG, EXPE, QUAL have 5 indicators (C(5,4) = 5 tetrads each);
    # VAL, SAT, LOY have 4 indicators (C(4,4) = 1 tetrad each).
    # Total = 3 * 5 + 3 * 1 = 18 tetrads, 6 blocks.
    assert len(summary) == 6
    assert int(summary["n_tetrads"].sum()) == 18
    assert len(tetrads) == 18


def test_cta_skips_mode_b_blocks():
    """Mode B is formative — CTA-PLS is not meaningful there."""
    cta = _satisfaction_plspm(mode=Mode.B).cta(n_boot=100, seed=0)
    assert cta.tetrads().empty
    assert cta.summary().empty


def test_cta_skips_blocks_with_fewer_than_four_indicators():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["SAT"])
    config = c.Config(structure.path(), scaled=False)
    # IMAG: 3 indicators only (< 4), should be skipped
    config.add_lv("IMAG", Mode.A, MV("imag1"), MV("imag2"), MV("imag3"))
    # SAT: 4 indicators, should be tested (C(4,4) = 1 tetrad)
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    plspm_calc = Plspm(satisfaction, config, Scheme.CENTROID)

    cta = plspm_calc.cta(n_boot=100, seed=0)
    assert set(cta.summary()["lv"]) == {"SAT"}
    assert int(cta.summary()["n_tetrads"].iloc[0]) == 1


def test_cta_supports_reflective_for_one_factor_data():
    """Single-common-factor data — every model-implied tetrad has expectation
    zero, so CTA-PLS should fail to reject reflective specification."""
    data = _one_factor_data(n=500, p=5, loadings=0.8, seed=0)
    cta = _cta_on_single_block(data, n_boot=300, alpha=0.05, seed=0)
    block = cta.summary().iloc[0]
    assert block["decision"] == "reflective supported"
    assert int(block["n_rejected"]) == 0


def test_cta_rejects_reflective_for_two_factor_data():
    """Two-factor block — tetrads should not vanish; CTA-PLS should reject
    the reflective specification."""
    data = _formative_like_data(n=600, p=6, seed=2)
    cta = _cta_on_single_block(data, n_boot=400, alpha=0.05, seed=0)
    block = cta.summary().iloc[0]
    assert block["decision"] == "reflective rejected"
    assert int(block["n_rejected"]) >= 1


def test_cta_p_values_are_floored_at_one_over_n_boot():
    """Even when the bootstrap never crosses zero, p ≥ 1/n_boot."""
    data = _formative_like_data(n=400, p=4, seed=3)
    n_boot = 200
    cta = _cta_on_single_block(data, n_boot=n_boot, seed=0)
    assert (cta.tetrads()["p_value"] >= 1.0 / n_boot - 1e-12).all()


def test_cta_is_deterministic_with_fixed_seed():
    data = _one_factor_data(n=300, p=5, loadings=0.8, seed=0)
    a = _cta_on_single_block(data, n_boot=120, seed=7).tetrads()
    b = _cta_on_single_block(data, n_boot=120, seed=7).tetrads()
    pd.testing.assert_frame_equal(a, b)


def test_cta_rejects_invalid_arguments():
    data = _one_factor_data(n=100, p=4, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        _cta_on_single_block(data, n_boot=10)
    with pytest.raises(ValueError, match="alpha"):
        _cta_on_single_block(data, alpha=1.5)
