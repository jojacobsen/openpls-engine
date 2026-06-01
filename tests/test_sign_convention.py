#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression test for #113: the per-LV sign vote in `_MetricWeights.calculate`
must only count indicators that actually belong to the LV.

Before the fix, every non-belonging cell in `cor * odm` was zero and
`math.copysign(1.0, 0)` evaluated to +1, so a 3-indicator LV embedded in a
17-indicator model carried ~14 phantom +1 votes and could never trigger a
flip — leaving SmartPLS-incompatible signs on small constructs (the OI
validation case)."""

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _two_lv_dataset_with_negative_block(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    big = rng.standard_normal(n)
    small = 0.5 * big + 0.5 * rng.standard_normal(n)
    frame = {f"b{i}": big + rng.standard_normal(n) * 0.3 for i in range(1, 15)}
    # SMALL has 3 indicators, all coded with an inverted scale (i.e. they
    # correlate negatively with the LV's "true" direction). Without the fix,
    # the 14 phantom +1 votes from the BIG block out-vote SMALL's -3, so the
    # engine leaves SMALL on the wrong sign.
    frame["s1"] = -small + rng.standard_normal(n) * 0.3
    frame["s2"] = -small + rng.standard_normal(n) * 0.3
    frame["s3"] = -small + rng.standard_normal(n) * 0.3
    return pd.DataFrame(frame)


def test_small_lv_sign_is_independent_of_unrelated_lv_size():
    data = _two_lv_dataset_with_negative_block()

    structure = c.Structure()
    structure.add_path(["BIG"], ["SMALL"])
    config = c.Config(structure.path())
    config.add_lv_with_columns_named("BIG", Mode.A, data, "b")
    config.add_lv_with_columns_named("SMALL", Mode.A, data, "s")

    result = Plspm(data, config, Scheme.PATH)
    outer = result.outer_model()

    # SMALL's indicators are uniformly coded negative w.r.t. the latent direction.
    # The post-convergence sign vote must reflect the SMALL indicators only — not
    # the surrounding BIG block — so all three SMALL loadings end up positive
    # (the engine flips the LV score during finalisation).
    for mv in ("s1", "s2", "s3"):
        assert outer.loc[mv, "loading"] > 0, (
            f"SMALL loading {mv} = {outer.loc[mv, 'loading']:.3f}; "
            "expected positive after per-LV sign-vote masking"
        )
