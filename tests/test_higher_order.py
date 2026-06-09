#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob / OpenPLS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.config import MV
from openpls.higher_order import HigherOrder
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _hoc_dataset(
    n: int = 600,
    loading: float = 0.8,
    hoc_to_y: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a 3-first-order + 1-target dataset.

    The three first-order factors share a common HOC eta_h. The target
    Y is driven by eta_h directly. With ``loading`` close to 1 the
    first-order LVs are excellent reflectors of eta_h and the HOC path
    to Y must approach ``hoc_to_y``.
    """
    rng = np.random.default_rng(seed)
    eta_h = rng.standard_normal(n)
    eta_y = hoc_to_y * eta_h + rng.standard_normal(n) * np.sqrt(1 - hoc_to_y ** 2)

    def block(prefix: str, p: int, factor: np.ndarray) -> pd.DataFrame:
        err = rng.standard_normal((n, p)) * np.sqrt(1 - loading ** 2)
        arr = loading * factor[:, None] + err
        return pd.DataFrame(arr, columns=[f"{prefix}{i+1}" for i in range(p)])

    # First-order LVs all reflect eta_h with small idiosyncratic shifts so
    # they are clearly distinct constructs but driven by the same higher
    # factor (the canonical HOC setup).
    eta1 = eta_h + 0.2 * rng.standard_normal(n)
    eta2 = eta_h + 0.2 * rng.standard_normal(n)
    eta3 = eta_h + 0.2 * rng.standard_normal(n)
    df = pd.concat(
        [
            block("x1_", 4, eta1),
            block("x2_", 4, eta2),
            block("x3_", 4, eta3),
            block("y", 3, eta_y),
        ],
        axis=1,
    )
    return df


def _stage1_config(df: pd.DataFrame, first_order_mode: Mode = Mode.A) -> c.Config:
    structure = c.Structure()
    structure.add_path(["X1"], ["Y"])
    structure.add_path(["X2"], ["Y"])
    structure.add_path(["X3"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("X1", first_order_mode, df, "x1_")
    config.add_lv_with_columns_named("X2", first_order_mode, df, "x2_")
    config.add_lv_with_columns_named("X3", first_order_mode, df, "x3_")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")
    return config


def _stage2_structure() -> c.Structure:
    structure = c.Structure()
    structure.add_path(["HOC"], ["Y"])
    return structure


def test_higher_order_type_i_reflective_reflective():
    df = _hoc_dataset(seed=1)
    base = Plspm(df, _stage1_config(df, Mode.A), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    # HOC loadings on its first-order indicators should all be strong
    # and positive: a Type-I HOC where the first-order LVs are excellent
    # reflectors of the same higher factor.
    loadings = hoc.loadings()
    assert (loadings > 0.7).all()
    # HOC path to Y should be in the right direction and substantial.
    paths = hoc.path_coefficients()
    assert paths.loc["Y", "HOC"] > 0.3


def test_higher_order_type_ii_reflective_formative():
    df = _hoc_dataset(seed=2)
    base = Plspm(df, _stage1_config(df, Mode.A), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.B,
        structure=_stage2_structure(),
    )
    assert hoc.hoc_mode() == Mode.B
    # Mode B HOC: formative weights, not loadings. summary() reports them.
    summary = hoc.summary()
    assert "weight" in summary.columns
    assert (summary["weight"] > 0).all()


def test_higher_order_type_iii_formative_reflective():
    df = _hoc_dataset(seed=3)
    base = Plspm(df, _stage1_config(df, Mode.B), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    loadings = hoc.loadings()
    # First-order LVs are formative composites of the HOC indicators —
    # they should still show consistent direction onto the HOC.
    assert (loadings > 0).all()


def test_higher_order_type_iv_formative_formative():
    df = _hoc_dataset(seed=4)
    base = Plspm(df, _stage1_config(df, Mode.B), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.B,
        structure=_stage2_structure(),
    )
    summary = hoc.summary()
    assert "weight" in summary.columns


def test_higher_order_stage2_has_hoc_and_drops_first_order():
    df = _hoc_dataset(seed=5)
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    refit = hoc.refit()
    lvs = list(refit.path_coefficients().index)
    assert "HOC" in lvs
    assert "Y" in lvs
    for old in ["X1", "X2", "X3"]:
        assert old not in lvs


def test_higher_order_recovers_hoc_path_close_to_truth():
    """With strong loadings and a known underlying eta_h → Y coefficient,
    the disjoint two-stage HOC path estimate should land near the truth."""
    truth = 0.7
    df = _hoc_dataset(n=1500, loading=0.9, hoc_to_y=truth, seed=42)
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    estimate = float(hoc.path_coefficients().loc["Y", "HOC"])
    assert abs(estimate - truth) < 0.05


def test_higher_order_rejects_duplicate_first_order():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1", "X1"],
            mode=Mode.A,
            structure=_stage2_structure(),
        )


def test_higher_order_rejects_unknown_first_order():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1", "NOT_IN_MODEL"],
            mode=Mode.A,
            structure=_stage2_structure(),
        )


def test_higher_order_rejects_name_collision():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    with pytest.raises(ValueError):
        base.higher_order(
            name="X1",  # already an LV
            first_order=["X1", "X2", "X3"],
            mode=Mode.A,
            structure=_stage2_structure(),
        )


def test_higher_order_rejects_first_order_in_stage2_structure():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    bad = c.Structure()
    bad.add_path(["HOC"], ["Y"])
    bad.add_path(["X1"], ["Y"])  # first-order LV in stage-2 — must reject
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1", "X2", "X3"],
            mode=Mode.A,
            structure=bad,
        )


def test_higher_order_rejects_unknown_stage2_lv():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    bad = c.Structure()
    bad.add_path(["HOC"], ["NEW_LV"])  # not in base config
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1", "X2", "X3"],
            mode=Mode.A,
            structure=bad,
        )


def test_higher_order_requires_hoc_in_structure():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    bad = c.Structure()
    bad.add_path(["Y"], ["X2"])  # any path that does not mention HOC
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1", "X2", "X3"],
            mode=Mode.A,
            structure=bad,
        )


def test_higher_order_requires_two_first_order():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    with pytest.raises(ValueError):
        base.higher_order(
            name="HOC",
            first_order=["X1"],
            mode=Mode.A,
            structure=_stage2_structure(),
        )


def test_higher_order_base_and_refit_are_distinct_fits():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    assert hoc.base() is base
    assert hoc.refit() is not base


def test_higher_order_indicator_columns_use_namespaced_names():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    cols = hoc.indicator_columns()
    assert cols == {"X1": "HOC__X1", "X2": "HOC__X2", "X3": "HOC__X3"}


def test_higher_order_summary_shape_and_columns_mode_a():
    df = _hoc_dataset()
    base = Plspm(df, _stage1_config(df), Scheme.CENTROID)
    hoc = base.higher_order(
        name="HOC",
        first_order=["X1", "X2", "X3"],
        mode=Mode.A,
        structure=_stage2_structure(),
    )
    summary = hoc.summary()
    assert list(summary.columns) == ["first_order", "loading", "stage1_r_squared"]
    assert list(summary["first_order"]) == ["X1", "X2", "X3"]
    # X1/X2/X3 are exogenous in stage 1 → R² reported as 0 by inner_summary.
    assert (summary["stage1_r_squared"] == 0.0).all()


def test_higher_order_can_chain_nested_hoc():
    """Calling higher_order on the stage-2 refit should chain — a HOC
    can in turn become a first-order of a third-order construct."""
    # Build a model with two sibling HOCs both pointing at Y.
    df = _hoc_dataset(seed=11)
    # Add a fourth first-order block so we can group (X1,X2) into HOC_A
    # and (X3,X4) into HOC_B.
    rng = np.random.default_rng(13)
    eta = df["x1_1"] * 0.6 + rng.standard_normal(len(df)) * 0.4
    err = rng.standard_normal((len(df), 4)) * np.sqrt(1 - 0.7 ** 2)
    block_4 = pd.DataFrame(
        0.7 * eta.to_numpy()[:, None] + err,
        columns=[f"x4_{i+1}" for i in range(4)],
        index=df.index,
    )
    df2 = pd.concat([df, block_4], axis=1)
    structure = c.Structure()
    structure.add_path(["X1"], ["Y"])
    structure.add_path(["X2"], ["Y"])
    structure.add_path(["X3"], ["Y"])
    structure.add_path(["X4"], ["Y"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["X1", "X2", "X3", "X4"]:
        config.add_lv_with_columns_named(lv, Mode.A, df2, lv.lower() + "_")
    config.add_lv_with_columns_named("Y", Mode.A, df2, "y")
    base = Plspm(df2, config, Scheme.CENTROID)

    # The disjoint API rolls one HOC at a time. We do HOC_A first into
    # an intermediate stage 2 that still keeps X3 and X4 as siblings.
    s2a = c.Structure()
    s2a.add_path(["HOC_A"], ["Y"])
    s2a.add_path(["X3"], ["Y"])
    s2a.add_path(["X4"], ["Y"])
    stage2 = base.higher_order(
        name="HOC_A", first_order=["X1", "X2"], mode=Mode.A, structure=s2a
    ).refit()
    # Then a second HOC on top of the stage-2 fit.
    s3 = c.Structure()
    s3.add_path(["HOC_A"], ["Y"])
    s3.add_path(["HOC_B"], ["Y"])
    stage3 = stage2.higher_order(
        name="HOC_B", first_order=["X3", "X4"], mode=Mode.A, structure=s3
    )
    paths = stage3.path_coefficients()
    assert "HOC_A" in paths.index
    assert "HOC_B" in paths.index
