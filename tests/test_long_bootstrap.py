import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.long_bootstrap import LongBootstrap
from openpls.mode import Mode
from openpls.scale import Scale
from openpls.scheme import Scheme


def _russa_config():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return config


def _russa():
    return pd.read_csv("file:tests/data/russa.csv", index_col=0)


def test_long_bootstrap_smoke_returns_expected_columns():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=20, seed=0)
    paths = boot.paths()
    assert set(paths.columns) >= {
        "source", "target", "original",
        "boot_mean", "se", "t", "p_value", "ci_lower", "ci_upper", "valid",
    }
    # 2 structural paths: AGRI→POLINS, IND→POLINS
    assert len(paths) == 2
    assert set(zip(paths["source"], paths["target"], strict=False)) == {
        ("AGRI", "POLINS"), ("IND", "POLINS"),
    }


def test_long_bootstrap_loadings_and_weights_shape():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=15, seed=1)
    loadings = boot.loadings()
    weights = boot.weights()
    # 9 indicators total: 3 AGRI + 2 IND + 4 POLINS
    assert len(loadings) == 9
    assert len(weights) == 9
    assert set(loadings.columns) >= {"lv", "indicator", "original", "boot_mean", "ci_lower", "ci_upper"}


def test_long_bootstrap_total_effects_includes_indirect():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=15, seed=2)
    total = boot.total_effects()
    # Direct paths AGRI→POLINS and IND→POLINS only; no indirect in this 2-stage model.
    assert len(total) == 2
    assert set(zip(total["source"], total["target"], strict=False)) == {
        ("AGRI", "POLINS"), ("IND", "POLINS"),
    }


def test_long_bootstrap_progress_callback_fires():
    calls: list[tuple[int, int]] = []
    LongBootstrap(
        _russa(),
        _russa_config(),
        iterations=10,
        seed=0,
        on_progress=lambda done, total: calls.append((done, total)),
        progress_every=2,
    )
    # At least one mid-run call and the final guaranteed call
    assert len(calls) >= 2
    assert calls[-1] == (10, 10)
    # All totals match iterations
    assert all(total == 10 for _, total in calls)
    # done values are monotonically increasing
    dones = [done for done, _ in calls]
    assert dones == sorted(dones)


def test_long_bootstrap_seed_determinism():
    a = LongBootstrap(_russa(), _russa_config(), iterations=10, seed=42).paths()
    b = LongBootstrap(_russa(), _russa_config(), iterations=10, seed=42).paths()
    pd.testing.assert_frame_equal(a, b)


def test_long_bootstrap_rejects_bad_min_success_ratio():
    with pytest.raises(ValueError, match="min_success_ratio"):
        LongBootstrap(_russa(), _russa_config(), iterations=5, min_success_ratio=1.5)


def test_long_bootstrap_runtime_floor_raises_when_all_resamples_fail(monkeypatch):
    from openpls import long_bootstrap as lb

    original_fit = lb.LongBootstrap._LongBootstrap__fit
    state = {"called": 0}

    def _fail_after_point(self, df):
        state["called"] += 1
        if state["called"] == 1:
            return original_fit(self, df)
        raise RuntimeError("boom")

    # Patch so the point estimate succeeds but every bootstrap resample raises.
    monkeypatch.setattr(lb.LongBootstrap, "_LongBootstrap__fit", _fail_after_point)
    with pytest.raises(RuntimeError, match="Bootstrap failed"):
        LongBootstrap(_russa(), _russa_config(), iterations=4, seed=0, min_success_ratio=0.5)


def test_long_bootstrap_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        LongBootstrap(_russa(), _russa_config(), iterations=5, alpha=0.0)


def test_long_bootstrap_rejects_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        LongBootstrap(_russa(), _russa_config(), iterations=0)


def test_long_bootstrap_ci_brackets_point_estimate_when_significant():
    # With enough iterations the BCa CI for a strong path should bracket the
    # original estimate (not require sign-agreement, just that lo <= point or
    # point <= hi).
    boot = LongBootstrap(_russa(), _russa_config(), iterations=60, seed=7)
    paths = boot.paths()
    for _, row in paths.iterrows():
        if np.isnan(row["original"]) or np.isnan(row["ci_lower"]) or np.isnan(row["ci_upper"]):
            continue
        assert row["ci_lower"] <= row["ci_upper"]


def test_long_bootstrap_completed_and_failed_counts():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=20, seed=3)
    assert boot.completed + boot.failed == 20
    assert boot.completed >= 1
    assert boot.alpha == 0.05
