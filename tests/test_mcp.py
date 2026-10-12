"""Tests for the MCP tool compute helpers (openpls.mcp.core).

These exercise the pure logic without the MCP SDK, so they run in the base
test env (no `[mcp]` extra needed).
"""

import math

import pytest

from openpls.mcp import core

DATA = "tests/data/satisfaction.csv"

ECSI_LVS = [
    {"name": "IMAG", "indicators": ["imag1", "imag2", "imag3"], "mode": "A"},
    {"name": "EXPE", "indicators": ["expe1", "expe2", "expe3"], "mode": "A"},
    {"name": "QUAL", "indicators": ["qual1", "qual2", "qual3"], "mode": "A"},
    {"name": "VAL", "indicators": ["val1", "val2"], "mode": "A"},
    {"name": "SAT", "indicators": ["sat1", "sat2", "sat3"], "mode": "A"},
    {"name": "LOY", "indicators": ["loy1", "loy2", "loy3"], "mode": "A"},
]
ECSI_PATHS = [
    {"source": "IMAG", "target": "EXPE"},
    {"source": "IMAG", "target": "SAT"},
    {"source": "IMAG", "target": "LOY"},
    {"source": "EXPE", "target": "QUAL"},
    {"source": "EXPE", "target": "VAL"},
    {"source": "EXPE", "target": "SAT"},
    {"source": "QUAL", "target": "VAL"},
    {"source": "QUAL", "target": "SAT"},
    {"source": "VAL", "target": "SAT"},
    {"source": "SAT", "target": "LOY"},
]


def test_describe_dataset():
    info = core.describe_dataset(DATA)
    assert info["rows"] == 250
    names = {c["name"] for c in info["columns"]}
    assert {"imag1", "sat1", "loy3"} <= names
    assert all(c["type"] == "numeric" for c in info["columns"] if c["name"].startswith("sat"))


def test_fit_pls_sem_ecsi():
    res = core.fit_pls_sem(DATA, ECSI_LVS, ECSI_PATHS)

    # Goodness-of-fit and model fit present and sane.
    assert 0.0 < res["goodness_of_fit"] < 1.0
    assert 0.0 < res["model_fit"]["srmr"] < 0.2

    # R² for Satisfaction matches the known ECSI reference (~0.70).
    r2 = {row["lv"]: row["r_squared"] for row in res["r_squared"]}
    assert math.isclose(r2["SAT"], 0.697, abs_tol=0.01)
    assert r2["IMAG"] == 0.0  # exogenous

    # Path IMAG -> EXPE is reported with a significant estimate.
    paths = {(p["source"], p["target"]): p for p in res["path_coefficients"]}
    assert math.isclose(paths[("IMAG", "EXPE")]["estimate"], 0.453, abs_tol=0.02)
    assert paths[("IMAG", "EXPE")]["p_value"] < 0.05

    # Reliability panel carries alpha + rho_c + AVE per construct.
    rel = {row["lv"]: row for row in res["reliability"]}
    assert 0.8 < rel["SAT"]["rho_c"] <= 1.0
    assert 0.5 < rel["SAT"]["ave"] <= 1.0


def test_fit_rejects_unknown_indicator():
    bad = [{"name": "X", "indicators": ["does_not_exist"], "mode": "A"}]
    with pytest.raises(ValueError, match="not in dataset"):
        core.fit_pls_sem(DATA, bad, [{"source": "X", "target": "X"}])


def test_fit_rejects_path_to_unknown_lv():
    with pytest.raises(ValueError, match="unknown LV"):
        core.fit_pls_sem(DATA, ECSI_LVS, [{"source": "IMAG", "target": "NOPE"}])
