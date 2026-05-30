import math

import numpy as np
import pandas as pd
import pytest

import plspm.config as c
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme


def _russa_path():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    return structure.path()


def _russa_config():
    config = c.Config(_russa_path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return config


def test_default_is_casewise_unchanged():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    a = Plspm(russa, _russa_config(), Scheme.CENTROID, 100, 1e-7)
    b = Plspm(russa, _russa_config(), Scheme.CENTROID, 100, 1e-7, missing_strategy="casewise")
    pd.testing.assert_frame_equal(a.scores(), b.scores())


def test_mean_strategy_changes_results_vs_casewise_on_missing():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    russa_missing = russa.astype(float).copy()
    russa_missing.iloc[0, 0] = np.nan
    russa_missing.iloc[3, 3] = np.nan
    russa_missing.iloc[5, 5] = np.nan

    plspm_mean = Plspm(
        russa_missing, _russa_config(), Scheme.CENTROID, 100, 1e-7, missing_strategy="mean"
    )
    plspm_casewise = Plspm(
        russa_missing, _russa_config(), Scheme.CENTROID, 100, 1e-7, missing_strategy="casewise"
    )

    # Mean strategy fills the NaN cells before estimation, so the scores diverge
    # from the casewise result that leaves NaN in place.
    assert (plspm_mean.scores() - plspm_casewise.scores()).abs().max().max() > 1e-3


def test_mean_strategy_noop_on_clean_data():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    plspm_default = Plspm(russa, _russa_config(), Scheme.CENTROID, 100, 1e-7)
    plspm_mean = Plspm(
        russa, _russa_config(), Scheme.CENTROID, 100, 1e-7, missing_strategy="mean"
    )
    pd.testing.assert_frame_equal(plspm_default.scores(), plspm_mean.scores())


def test_invalid_strategy_raises():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    with pytest.raises(ValueError):
        Plspm(russa, _russa_config(), missing_strategy="nope")


def test_mean_replacement_matches_manual():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0).astype(float)
    russa.loc[russa.index[0], "gini"] = np.nan
    russa.loc[russa.index[2], "rent"] = np.nan

    expected = russa.copy()
    expected["gini"] = expected["gini"].fillna(expected["gini"].mean())
    expected["rent"] = expected["rent"].fillna(expected["rent"].mean())

    plspm_mean = Plspm(
        russa, _russa_config(), Scheme.CENTROID, 100, 1e-7, missing_strategy="mean"
    )
    plspm_expected = Plspm(expected, _russa_config(), Scheme.CENTROID, 100, 1e-7)

    pd.testing.assert_frame_equal(plspm_mean.scores(), plspm_expected.scores())
    # also: AVE should match because we standardize after imputation
    np_close = math.isclose(
        float(plspm_mean.inner_summary().loc["POLINS", "ave"]),
        float(plspm_expected.inner_summary().loc["POLINS", "ave"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    assert np_close
