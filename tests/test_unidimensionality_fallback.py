import math

import numpy as np
import pandas as pd

import plspm.config as c
from plspm.mode import Mode
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme


def _russa_path():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    return structure.path()


def _russa_config_mode_a():
    config = c.Config(_russa_path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return config


def test_clean_data_reliability_finite():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    plspm_calc = Plspm(russa, _russa_config_mode_a(), Scheme.CENTROID, 100, 1e-7)
    unidim = plspm_calc.unidimensionality()
    for lv in ("AGRI", "IND", "POLINS"):
        assert math.isfinite(float(unidim.loc[lv, "dillon_goldstein_rho"]))
        assert math.isfinite(float(unidim.loc[lv, "eig_1st"]))


def test_fallback_fires_on_missing_values():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    russa.iloc[0, 0] = np.nan  # gini
    russa.iloc[3, 3] = np.nan  # gnpr/labo column
    russa.iloc[5, 5] = np.nan  # demo/inst column

    plspm_calc = Plspm(russa, _russa_config_mode_a(), Scheme.CENTROID, 100, 1e-7)
    unidim = plspm_calc.unidimensionality()

    # Every Mode A LV should now report finite reliability + eigenvalues
    for lv in ("AGRI", "IND", "POLINS"):
        for col in ("dillon_goldstein_rho", "eig_1st"):
            val = float(unidim.loc[lv, col])
            assert math.isfinite(val), f"{lv}/{col} should be finite, got {val}"


def test_mode_b_clean_data_alpha_rho_nan():
    russa = pd.read_csv("file:tests/data/russa.csv", index_col=0)
    config = c.Config(_russa_path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.B, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.B, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.B, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))

    plspm_calc = Plspm(russa, config, Scheme.CENTROID, 100, 1e-7)
    unidim = plspm_calc.unidimensionality()

    # alpha and rho are reflective-only — Mode B should keep them NaN
    for lv in ("AGRI", "IND", "POLINS"):
        assert pd.isna(unidim.loc[lv, "cronbach_alpha"]), lv
        assert pd.isna(unidim.loc[lv, "dillon_goldstein_rho"]), lv
        # eigenvalues are still defined and finite
        assert math.isfinite(float(unidim.loc[lv, "eig_1st"])), lv
