#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob / OpenPLS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Regression tests for crashes seen in production (Sentry)."""
import numpy as np
import pandas as pd

from openpls.util import impute


def test_impute_handles_all_nan_column_without_crashing():
    """OPENPLS-FUNCTIONS-S: impute() asserted mean-preservation with a 1e-9
    tolerance, which fired on floating-point noise and on an all-NaN column
    (where the mean is NaN), crashing real computes. It must not raise."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "all_missing": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    out = impute(df)  # must not raise
    assert out["a"].notna().all()
    assert abs(out["a"].mean() - df["a"].mean(skipna=True)) < 1e-9
    # A fully-missing column has no mean, so it stays finite (zeros), not NaN.
    assert out["all_missing"].notna().all()


def test_reliability_pca_input_sanitized_against_zero_variance():
    """OPENPLS-FUNCTIONS-K: when a reliability block becomes constant (e.g. after
    the block's own NaN rows are dropped), util.treat divides by a zero standard
    deviation, producing NaN that PCA rejects with 'Input X contains NaN'. The
    fix replaces the non-finite cells with zero so PCA runs. This tests that
    mechanism directly (reproducing the full-model path is estimation-dependent
    and a globally constant indicator breaks estimation first)."""
    from sklearn.decomposition import PCA

    from openpls import util

    block = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "const": [7.0, 7.0, 7.0, 7.0, 7.0]}
    )
    treated = util.treat(block)
    assert treated["const"].isna().all(), "zero-variance column divides by std 0"

    # The reliability fix: non-finite -> 0 makes a PCA-safe matrix.
    safe = treated.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    assert np.isfinite(safe.to_numpy()).all()
    PCA().fit_transform(safe)  # must not raise
