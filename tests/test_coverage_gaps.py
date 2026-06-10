"""Targeted tests for coverage gaps.

These tests fill specific paths that the broader regression and case tests
do not naturally hit: property getters, early-return defensive branches,
and the BootstrapProcess worker (whose run() body is otherwise only
exercised inside a forked subprocess and therefore invisible to coverage).
"""

from __future__ import annotations

from multiprocessing import Queue

import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.bootstrap import BootstrapProcess, _create_summary
from openpls.estimator import Estimator
from openpls.htmt import HTMT
from openpls.htmt2 import HTMT2
from openpls.inner_model import InnerModel
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.predict import PLSPredict
from openpls.q_squared import QSquared
from openpls.scheme import Scheme
from openpls.weights import WeightsCalculatorFactory


def _satisfaction_config():
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(structure.path(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return config, satisfaction


def test_bootstrap_worker_run_in_main_thread():
    """Run BootstrapProcess.run() directly (no subprocess) so coverage tracks it."""
    config, data = _satisfaction_config()
    filtered = config.filter(data)
    fit = Plspm(data, config, Scheme.CENTROID)
    inner = InnerModel(config.path(), fit.scores())
    correction = float(np.sqrt(filtered.shape[0] / (filtered.shape[0] - 1)))
    calculator = WeightsCalculatorFactory(config, 100, 1e-7, correction, Scheme.CENTROID)

    queue: Queue = Queue()
    worker = BootstrapProcess(queue, config, filtered, inner, calculator, iterations=3)
    worker.run()

    results = queue.get(timeout=10)
    assert set(results.keys()) == {"weights", "r_squared", "total_effects", "paths", "loadings"}
    assert not results["weights"].empty
    assert not results["r_squared"].empty


def test_bootstrap_create_summary_columns():
    """_create_summary returns the standard six-column bootstrap panel."""
    samples = pd.DataFrame(np.random.RandomState(0).randn(50, 3), columns=["a", "b", "c"])
    original = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    summary = _create_summary(samples, original)
    assert list(summary.columns) == ["original", "mean", "std.error", "perc.025", "perc.975", "t stat."]
    assert list(summary.index) == ["a", "b", "c"]


def test_predict_k_and_repeats_properties():
    config, data = _satisfaction_config()
    fit = Plspm(data, config, Scheme.CENTROID)
    pred = fit.predict(k=4, repeats=2, seed=7)
    assert pred.k == 4
    assert pred.repeats == 2


def test_predict_invalid_k_raises():
    config, data = _satisfaction_config()
    with pytest.raises(ValueError, match="k must be"):
        PLSPredict(config, data, k=1)


def test_predict_invalid_repeats_raises():
    config, data = _satisfaction_config()
    with pytest.raises(ValueError, match="repeats must be"):
        PLSPredict(config, data, k=5, repeats=0)


def test_predict_k_larger_than_n_raises():
    config, data = _satisfaction_config()
    small = data.iloc[:3]
    with pytest.raises(ValueError, match="cannot exceed sample size"):
        PLSPredict(config, small, k=5)


def test_qsquared_omission_distance_property():
    config, data = _satisfaction_config()
    qs = QSquared(config, data, Scheme.CENTROID, omission_distance=8)
    assert qs.omission_distance == 8


def test_qsquared_invalid_omission_distance_raises():
    config, data = _satisfaction_config()
    with pytest.raises(ValueError, match="omission_distance"):
        QSquared(config, data, Scheme.CENTROID, omission_distance=1)


def _single_lv_path():
    """Manually build a 1×1 path matrix to bypass Structure's cycle check."""
    return pd.DataFrame([[0]], index=["A"], columns=["A"])


def test_htmt_handles_single_lv_block():
    """HTMT with one LV exercises the early-return branch (len(lv_names) < 2)."""
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    cfg = c.Config(_single_lv_path())
    cfg.add_lv_with_columns_named("A", Mode.A, satisfaction, "imag")
    htmt = HTMT(cfg, satisfaction)
    m = htmt.matrix()
    assert m.shape == (1, 1)
    assert pd.isna(m.iloc[0, 0])


def test_htmt2_handles_single_lv_block():
    """Same early-return path on HTMT2."""
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    cfg = c.Config(_single_lv_path())
    cfg.add_lv_with_columns_named("A", Mode.A, satisfaction, "imag")
    htmt2 = HTMT2(cfg, satisfaction)
    m = htmt2.matrix()
    assert m.shape == (1, 1)
    assert pd.isna(m.iloc[0, 0])


def test_estimator_estimate_returns_three_pieces():
    """Estimator.estimate() returns (data, scores, weights) — sanity-check shape."""
    config, data = _satisfaction_config()
    filtered = config.filter(data)
    estimator = Estimator(config)
    correction = float(np.sqrt(filtered.shape[0] / (filtered.shape[0] - 1)))
    calculator = WeightsCalculatorFactory(config, 100, 1e-7, correction, Scheme.CENTROID)
    final_data, scores, weights = estimator.estimate(calculator, filtered)
    assert final_data.shape[0] == scores.shape[0]
    assert weights.shape[0] > 0
