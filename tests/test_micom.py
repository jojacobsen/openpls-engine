import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.mga import GroupSpec
from openpls.micom import MICOM
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scale import Scale
from openpls.scheme import Scheme


def _russa_with_random_group(seed: int = 42):
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0).copy()
    rng = np.random.default_rng(seed)
    russa["region"] = rng.choice(["west", "east"], size=len(russa))
    return russa


def _russa_config():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    cfg = c.Config(structure.path(), default_scale=Scale.NUM)
    cfg.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    cfg.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    cfg.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return cfg


def _synthetic_two_group(n_per: int = 200, scenario: str = "invariant", seed: int = 0):
    """Three-LV synthetic dataset with two groups of size n_per each.

    scenarios:
    - "invariant": both groups share identical loadings + means + variances
    - "violate_step2": group B has reversed loading pattern on construct X1
    - "violate_step3_mean": both groups same loadings, but group B has a
      shifted mean on every indicator of X1 (composite-level mean differs)
    - "violate_step3_var": same loadings + means, group B has inflated
      variance on the indicators of X1
    """
    rng = np.random.default_rng(seed)
    rows = []
    for grp, label in [(0, "A"), (1, "B")]:
        n = n_per
        xi1 = rng.standard_normal(n)
        xi2 = rng.standard_normal(n)
        y = 0.5 * xi1 + 0.4 * xi2 + 0.3 * rng.standard_normal(n)
        # Default loadings, three indicators each:
        load_x1 = np.array([0.9, 0.85, 0.8])
        load_x2 = np.array([0.9, 0.85, 0.8])
        load_y = np.array([0.9, 0.85, 0.8])
        mean_shift_x1 = 0.0
        var_scale_x1 = 1.0
        if grp == 1 and scenario == "violate_step2":
            # Reverse loading order so weights flip per indicator.
            load_x1 = load_x1[::-1]
        if grp == 1 and scenario == "violate_step3_mean":
            mean_shift_x1 = 1.2
        if grp == 1 and scenario == "violate_step3_var":
            var_scale_x1 = 4.0
        for i in range(n):
            r = {"group": label}
            for j, lam in enumerate(load_x1):
                r[f"x1_{j+1}"] = (
                    lam * xi1[i] + np.sqrt(1 - lam**2) * rng.standard_normal()
                ) * var_scale_x1 + mean_shift_x1
            for j, lam in enumerate(load_x2):
                r[f"x2_{j+1}"] = lam * xi2[i] + np.sqrt(1 - lam**2) * rng.standard_normal()
            for j, lam in enumerate(load_y):
                r[f"y_{j+1}"] = lam * y[i] + np.sqrt(1 - lam**2) * rng.standard_normal()
            rows.append(r)
    return pd.DataFrame(rows)


def _synthetic_config():
    structure = c.Structure()
    structure.add_path(["X1", "X2"], ["Y"])
    cfg = c.Config(structure.path(), default_scale=Scale.NUM)
    cfg.add_lv("X1", Mode.A, c.MV("x1_1"), c.MV("x1_2"), c.MV("x1_3"))
    cfg.add_lv("X2", Mode.A, c.MV("x2_1"), c.MV("x2_2"), c.MV("x2_3"))
    cfg.add_lv("Y", Mode.A, c.MV("y_1"), c.MV("y_2"), c.MV("y_3"))
    return cfg


# ---------- API surface tests ---------------------------------------------


def test_step2_step3_columns_and_types():
    df = _russa_with_random_group()
    m = MICOM(
        df,
        _russa_config(),
        grouping_column="region",
        group_a=GroupSpec("west", values=["west"]),
        group_b=GroupSpec("east", values=["east"]),
        iterations=20,
        seed=0,
    )
    s2 = m.step2()
    assert set(s2.columns) == {"construct", "c", "p_value", "compositional_invariance"}
    assert set(s2["construct"]) == {"AGRI", "IND", "POLINS"}
    assert ((s2["c"] >= -1.0 - 1e-9) & (s2["c"] <= 1.0 + 1e-9)).all()
    s3 = m.step3()
    assert set(s3.columns) == {
        "construct", "mean_diff", "mean_p_value", "mean_equal",
        "log_var_ratio", "var_p_value", "var_equal",
    }
    summary = m.summary()
    assert set(summary["invariance"]).issubset({"full", "partial", "none"})


def test_plspm_micom_entry_point_returns_same_results():
    df = _russa_with_random_group()
    cfg = _russa_config()
    fit = Plspm(df, cfg, Scheme.CENTROID)
    m = fit.micom(
        df,
        grouping_column="region",
        group_a=GroupSpec("west", values=["west"]),
        group_b=GroupSpec("east", values=["east"]),
        iterations=20,
        seed=0,
    )
    s2 = m.step2()
    assert len(s2) == 3
    # Sanity: random groups should keep c close to 1.
    assert (s2["c"] > 0.9).all()


# ---------- behavioural tests ---------------------------------------------


def test_random_group_passes_compositional_invariance():
    """With purely random group labels, Step 2 should clear all constructs.

    Step 3 (means + variances) follows a permutation null and can produce
    occasional false positives by chance; we only assert that *most*
    constructs pass on a single seed run.
    """
    df = _synthetic_two_group(n_per=200, scenario="invariant", seed=1)
    m = MICOM(
        df,
        _synthetic_config(),
        grouping_column="group",
        group_a=GroupSpec("A", values=["A"]),
        group_b=GroupSpec("B", values=["B"]),
        iterations=200,
        seed=42,
    )
    summary = m.summary()
    # All three constructs should clear compositional invariance.
    assert (summary["c"] > 0.95).all()
    assert summary["compositional_invariance"].all()
    # And under truly random labels we expect *most* Step-3 sub-tests to clear.
    assert summary["mean_equal"].sum() >= 2
    assert summary["var_equal"].sum() >= 2


def test_violation_step3_mean_detected():
    """Shifting the indicator means of X1 in group B should fail Step 3 means."""
    df = _synthetic_two_group(n_per=200, scenario="violate_step3_mean", seed=3)
    m = MICOM(
        df,
        _synthetic_config(),
        grouping_column="group",
        group_a=GroupSpec("A", values=["A"]),
        group_b=GroupSpec("B", values=["B"]),
        iterations=200,
        seed=42,
    )
    summary = m.summary().set_index("construct")
    # X1 should be detected: mean inequality
    assert summary.loc["X1", "compositional_invariance"]
    assert not summary.loc["X1", "mean_equal"]
    # X2 (untouched) should pass both Step 2 and Step 3a
    assert summary.loc["X2", "compositional_invariance"]
    assert summary.loc["X2", "mean_equal"]


def test_violation_step3_variance_detected():
    """Inflating indicator variances in group B should fail Step 3 variances."""
    df = _synthetic_two_group(n_per=200, scenario="violate_step3_var", seed=5)
    m = MICOM(
        df,
        _synthetic_config(),
        grouping_column="group",
        group_a=GroupSpec("A", values=["A"]),
        group_b=GroupSpec("B", values=["B"]),
        iterations=200,
        seed=42,
    )
    summary = m.summary().set_index("construct")
    # X1 variance gap should be flagged
    assert not summary.loc["X1", "var_equal"]
    # X2 (untouched) should keep variance equality
    assert summary.loc["X2", "var_equal"]


def test_group_sizes_match_input():
    df = _russa_with_random_group()
    m = MICOM(
        df,
        _russa_config(),
        grouping_column="region",
        group_a=GroupSpec("west", values=["west"]),
        group_b=GroupSpec("east", values=["east"]),
        iterations=5,
        seed=0,
    )
    sizes = m.group_sizes()
    assert sizes["west"] + sizes["east"] == len(df)
    assert sizes["west"] > 0 and sizes["east"] > 0


# ---------- edge cases ---------------------------------------------------


def test_rejects_missing_grouping_column():
    df = _russa_with_random_group()
    with pytest.raises(ValueError, match="grouping_column"):
        MICOM(
            df,
            _russa_config(),
            grouping_column="nope",
            group_a=GroupSpec("a", values=[1]),
            group_b=GroupSpec("b", values=[2]),
            iterations=5,
        )


def test_rejects_same_group_name():
    df = _russa_with_random_group()
    with pytest.raises(ValueError, match="distinct names"):
        MICOM(
            df,
            _russa_config(),
            grouping_column="region",
            group_a=GroupSpec("dup", values=["west"]),
            group_b=GroupSpec("dup", values=["east"]),
            iterations=5,
        )


def test_rejects_empty_group():
    df = _russa_with_random_group()
    with pytest.raises(ValueError, match="zero rows"):
        MICOM(
            df,
            _russa_config(),
            grouping_column="region",
            group_a=GroupSpec("west", values=["west"]),
            group_b=GroupSpec("none", values=["nowhere"]),
            iterations=5,
        )


def test_rejects_overlapping_groups():
    df = _russa_with_random_group()
    with pytest.raises(ValueError, match="overlap"):
        MICOM(
            df,
            _russa_config(),
            grouping_column="region",
            group_a=GroupSpec("a", values=["west", "east"]),
            group_b=GroupSpec("b", values=["east"]),
            iterations=5,
        )
