import numpy as np
import pandas as pd
import pytest

import plspm.config as c
from plspm.mga import MGA, GroupSpec
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme


def _russa_with_group():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    rng = np.random.default_rng(42)
    russa = russa.copy()
    russa["region"] = rng.choice(["west", "east"], size=len(russa))
    return russa


def _russa_config():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return config


def test_mga_per_group_estimates_match_individual_fits():
    russa = _russa_with_group()
    cfg = _russa_config()
    mga = MGA(
        russa,
        cfg,
        grouping_column="region",
        groups=[
            GroupSpec(name="west", values=["west"]),
            GroupSpec(name="east", values=["east"]),
        ],
        iterations=10,
        seed=0,
    )
    est = mga.group_estimates()
    assert set(est["group"].unique()) == {"west", "east"}
    assert set(est.columns) == {"group", "n", "source", "target", "estimate"}

    # Estimates should match a stand-alone Plspm fit on the same subset
    west_sub = russa.loc[russa["region"] == "west"].reset_index(drop=True)
    fit = Plspm(west_sub, _russa_config(), Scheme.CENTROID)
    expected = float(fit.path_coefficients().loc["POLINS", "AGRI"])
    actual = float(est.query("group == 'west' and source == 'AGRI' and target == 'POLINS'")
                      .iloc[0]["estimate"])
    assert abs(expected - actual) < 1e-9


def test_mga_comparisons_returns_pvalues():
    russa = _russa_with_group()
    cfg = _russa_config()
    mga = MGA(
        russa,
        cfg,
        grouping_column="region",
        groups=[
            GroupSpec(name="west", values=["west"]),
            GroupSpec(name="east", values=["east"]),
        ],
        iterations=20,
        seed=0,
    )
    comps = mga.comparisons()
    assert set(comps.columns) == {
        "groupA", "groupB", "source", "target",
        "estimateA", "estimateB", "difference", "p_value",
    }
    # With 20 permutations the minimum p-value (Phipson-Smyth) is (0+1)/(20+1) = 1/21
    assert (comps["p_value"] >= 1 / 21 - 1e-12).all()
    assert (comps["p_value"] <= 1.0).all()
    # Difference == estimateA - estimateB
    for _, row in comps.iterrows():
        assert abs(row["difference"] - (row["estimateA"] - row["estimateB"])) < 1e-9


def test_mga_three_groups_pairs():
    russa = _russa_with_group()
    # Re-label into three groups
    rng = np.random.default_rng(7)
    russa["region"] = rng.choice(["a", "b", "c"], size=len(russa))
    cfg = _russa_config()
    mga = MGA(
        russa,
        cfg,
        grouping_column="region",
        groups=[
            GroupSpec(name="a", values=["a"]),
            GroupSpec(name="b", values=["b"]),
            GroupSpec(name="c", values=["c"]),
        ],
        iterations=5,
        seed=0,
    )
    comps = mga.comparisons()
    # Three groups → C(3,2) = 3 pairs, each with 2 paths (AGRI→POLINS, IND→POLINS)
    assert len(comps) == 3 * 2
    assert set(zip(comps["groupA"], comps["groupB"], strict=False)) == {
        ("a", "b"), ("a", "c"), ("b", "c"),
    }


def test_mga_rejects_missing_column():
    russa = _russa_with_group()
    with pytest.raises(ValueError, match="grouping_column"):
        MGA(russa, _russa_config(), "no_such_col",
            [GroupSpec("a", values=[1]), GroupSpec("b", values=[2])])


def test_mga_rejects_empty_group():
    russa = _russa_with_group()
    with pytest.raises(ValueError, match="zero rows"):
        MGA(russa, _russa_config(), "region",
            [GroupSpec("west", values=["west"]),
             GroupSpec("nowhere", values=["does_not_exist"])])


def test_mga_numeric_range_group():
    russa = _russa_with_group()
    # Use gnpr (numeric) as grouping column
    cfg = _russa_config()
    mga = MGA(
        russa,
        cfg,
        grouping_column="gnpr",
        groups=[
            GroupSpec(name="low", range=(None, 6.0)),
            GroupSpec(name="high", range=(6.0, None)),
        ],
        iterations=5,
        seed=0,
    )
    est = mga.group_estimates()
    assert set(est["group"].unique()) == {"low", "high"}
