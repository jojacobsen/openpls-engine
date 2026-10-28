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

import openpls.config as c
from openpls.mode import Mode
from openpls.model_test import ModelTest, ModelTestMeasure
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _ecsi_config(df: pd.DataFrame) -> c.Config:
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=True)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, df, lv.lower())
    return config


def test_model_test_matches_csem_testomf_on_ecsi():
    """Observed discrepancies must match cSEM 0.7.1 ``testOMF`` on the ECSI
    common-factor model (the oracle), and the model must be rejected — the
    classic result where descriptive SRMR ≈ 0.087 looks borderline but the
    inferential test rejects."""
    df = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    mt = ModelTest(_ecsi_config(df), df, Scheme.PATH, n_boot=199, seed=1)
    res = mt.results()

    # cSEM targets (deterministic observed statistics).
    assert abs(res["srmr"].observed - 0.087007) < 0.002
    assert abs(res["d_uls"].observed - 2.861564) < 0.02
    assert abs(res["d_g"].observed - 1.154382) < 0.02

    # The strict test rejects this model on every measure.
    assert all(res[m].rejected for m in ("srmr", "d_uls", "d_g"))
    assert mt.rejected()
    # Inadmissible resamples are dropped, as cSEM does.
    assert mt.n_dropped() > 0
    assert mt.n_valid_resamples() > 0


def test_model_test_measure_shape_and_accessor():
    df = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    mt = Plspm(df, _ecsi_config(df), Scheme.PATH).model_test(n_boot=99, seed=1)
    summary = mt.summary()
    assert set(summary.index) == {"srmr", "d_uls", "d_g"}
    assert set(summary.columns) == {"observed", "hi95", "hi99", "p_value", "rejected"}
    for m in ("srmr", "d_uls", "d_g"):
        r = mt.results()[m]
        assert isinstance(r, ModelTestMeasure)
        assert r.observed > 0
        assert r.rejected == (r.observed > r.hi95)


def test_model_test_does_not_reject_a_fitting_model():
    """A saturated two-construct model (X -> Y, just-identified structurally)
    with clean reflective blocks fits: the OMF test should not reject."""
    rng = np.random.default_rng(3)
    n, p, loading = 500, 4, 0.8
    eta_x = rng.standard_normal(n)
    eta_y = 0.5 * eta_x + np.sqrt(1 - 0.25) * rng.standard_normal(n)
    x = loading * eta_x[:, None] + rng.standard_normal((n, p)) * np.sqrt(1 - loading**2)
    y = loading * eta_y[:, None] + rng.standard_normal((n, p)) * np.sqrt(1 - loading**2)
    df = pd.DataFrame(
        np.column_stack([x, y]),
        columns=[f"x{i + 1}" for i in range(p)] + [f"y{i + 1}" for i in range(p)],
    )
    structure = c.Structure()
    structure.add_path(["X"], ["Y"])
    config = c.Config(structure.path(), scaled=True)
    config.add_lv_with_columns_named("X", Mode.A, df, "x")
    config.add_lv_with_columns_named("Y", Mode.A, df, "y")

    mt = ModelTest(config, df, Scheme.PATH, n_boot=199, seed=1)
    assert not mt.rejected()


def test_model_test_is_deterministic_for_a_fixed_seed():
    """Same seed -> identical observed statistics and reference quantiles
    (the bootstrap must be reproducible, like the rest of the engine)."""
    df = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    a = ModelTest(_ecsi_config(df), df, Scheme.PATH, n_boot=99, seed=7).results()
    b = ModelTest(_ecsi_config(df), df, Scheme.PATH, n_boot=99, seed=7).results()
    for m in ("srmr", "d_uls", "d_g"):
        assert a[m].observed == b[m].observed
        assert a[m].hi95 == b[m].hi95
        assert a[m].hi99 == b[m].hi99


def test_model_test_exported_from_package():
    import openpls

    assert openpls.ModelTest is ModelTest
    assert openpls.ModelTestMeasure is ModelTestMeasure
