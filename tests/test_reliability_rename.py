"""Verify the 1.9.0 rename Unidimensionality -> Reliability is binary
compatible: the deprecated class and the deprecated Plspm.unidimensionality()
method still work and emit DeprecationWarning, and they return the same
data as the new Reliability / Plspm.reliability().
"""

import warnings

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.reliability import Reliability
from openpls.scheme import Scheme
from openpls.unidimensionality import Unidimensionality


def _satisfaction_plspm():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_plspm_unidimensionality_emits_deprecation_warning():
    fit = _satisfaction_plspm()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit.unidimensionality()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "Plspm.unidimensionality() must emit DeprecationWarning"
    assert "reliability" in str(deprecations[0].message).lower()


def test_plspm_unidimensionality_returns_same_data_as_reliability():
    fit = _satisfaction_plspm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = fit.unidimensionality()
    new = fit.reliability()
    pd.testing.assert_frame_equal(old, new)


def test_unidimensionality_class_emits_deprecation_warning():
    """Direct instantiation of the deprecated class must still work but
    emit DeprecationWarning."""
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE"])
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    correction = float(np.sqrt(satisfaction.shape[0] / (satisfaction.shape[0] - 1)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = Unidimensionality(config, satisfaction, correction)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "Unidimensionality must emit DeprecationWarning"

    new = Reliability(config, satisfaction, correction)
    pd.testing.assert_frame_equal(legacy.summary(), new.summary())
