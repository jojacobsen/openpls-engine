#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression test for #112: ECSI-style models in which a single-item LV
shares its name with the underlying indicator column must fit end-to-end."""

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _make_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "x1": latent + rng.standard_normal(n) * 0.4,
            "x2": latent + rng.standard_normal(n) * 0.4,
            "x3": latent + rng.standard_normal(n) * 0.4,
            "CUSCO": latent * 0.7 + rng.standard_normal(n) * 0.3,
        }
    )


def test_single_item_lv_with_matching_indicator_name():
    data = _make_dataset()

    structure = c.Structure()
    structure.add_path(["X"], ["CUSCO"])
    config = c.Config(structure.path())
    config.add_lv_with_columns_named("X", Mode.A, data, "x")
    config.add_lv("CUSCO", Mode.A, c.MV("CUSCO"))

    result = Plspm(data, config, Scheme.PATH)

    outer = result.outer_model()
    paths = result.path_coefficients()

    assert "X" in paths.index and "CUSCO" in paths.columns
    assert not outer["loading"].isna().any()
    # The single-item LV's loading on its sole indicator is 1.0 by construction.
    assert abs(outer.loc["CUSCO", "loading"] - 1.0) < 1e-9
    # The structural path from X to CUSCO is estimated and non-zero.
    assert abs(paths.loc["CUSCO", "X"]) > 0.5
