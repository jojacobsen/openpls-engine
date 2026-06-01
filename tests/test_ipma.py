import math

import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


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
    return Plspm(satisfaction, config, Scheme.CENTROID), satisfaction


def test_ipma_lv_table_has_expected_columns():
    fit, _ = _satisfaction_plspm()
    res = fit.ipma("SAT").latent_variables()
    assert set(res.columns) == {"importance", "performance"}
    assert res.index.name == "lv"


def test_ipma_only_includes_lvs_with_effect_on_target():
    fit, _ = _satisfaction_plspm()
    res = fit.ipma("SAT").latent_variables()
    # In ECSI, every LV except SAT itself has a (direct or indirect) effect on SAT.
    # Specifically IMAG, EXPE, QUAL, VAL feed into SAT. LOY does not.
    assert set(res.index) == {"IMAG", "EXPE", "QUAL", "VAL"}


def test_ipma_importance_matches_total_effects():
    fit, _ = _satisfaction_plspm()
    ipma_res = fit.ipma("SAT").latent_variables()
    effects = fit.effects()
    for lv in ipma_res.index:
        expected = float(effects[(effects["from"] == lv) & (effects["to"] == "SAT")]["total"].iloc[0])
        assert math.isclose(float(ipma_res.loc[lv, "importance"]), expected, abs_tol=1e-10)


def test_ipma_performance_within_0_100():
    fit, _ = _satisfaction_plspm()
    res = fit.ipma("SAT").latent_variables()
    for lv in res.index:
        perf = float(res.loc[lv, "performance"])
        assert math.isfinite(perf), f"{lv}: performance not finite"
        assert 0.0 <= perf <= 100.0, f"{lv}: performance {perf} outside [0, 100]"


def test_ipma_rejects_exogenous_target():
    fit, _ = _satisfaction_plspm()
    with pytest.raises(ValueError, match="exogenous"):
        fit.ipma("IMAG")


def test_ipma_rejects_unknown_target():
    fit, _ = _satisfaction_plspm()
    with pytest.raises(ValueError, match="not a latent variable"):
        fit.ipma("DOES_NOT_EXIST")


def test_ipma_scale_min_max_must_be_paired():
    fit, _ = _satisfaction_plspm()
    with pytest.raises(ValueError, match="together"):
        fit.ipma("SAT", scale_min=1.0)
    with pytest.raises(ValueError, match="together"):
        fit.ipma("SAT", scale_max=10.0)


def test_ipma_rejects_invalid_scale_bounds():
    fit, _ = _satisfaction_plspm()
    with pytest.raises(ValueError, match="greater"):
        fit.ipma("SAT", scale_min=10.0, scale_max=5.0)


def test_ipma_global_scale_changes_performance():
    fit, _ = _satisfaction_plspm()
    base = fit.ipma("SAT", scale_min=0.0, scale_max=10.0).latent_variables()
    # Widening the scale to 0-20 halves every rescaled value, so performance
    # must halve too.
    wide = fit.ipma("SAT", scale_min=0.0, scale_max=20.0).latent_variables()
    for lv in base.index:
        assert math.isclose(
            float(base.loc[lv, "performance"]),
            2.0 * float(wide.loc[lv, "performance"]),
            abs_tol=1e-9,
        ), f"{lv}: rescaling 0-10 vs 0-20 did not halve performance"


def test_ipma_indicator_scales_override_global():
    fit, _ = _satisfaction_plspm()
    base = fit.ipma("SAT", scale_min=0.0, scale_max=10.0).indicators()
    override = fit.ipma(
        "SAT",
        scale_min=0.0,
        scale_max=10.0,
        indicator_scales={"sat1": (0.0, 20.0)},
    ).indicators()
    assert math.isclose(float(base.loc[("SAT", "sat1"), "scale_max"]), 10.0)
    assert math.isclose(float(override.loc[("SAT", "sat1"), "scale_max"]), 20.0)
    # other indicators in SAT block still use the global bounds
    assert math.isclose(float(override.loc[("SAT", "sat2"), "scale_max"]), 10.0)


def test_ipma_indicators_table_structure():
    fit, _ = _satisfaction_plspm()
    ind = fit.ipma("SAT").indicators()
    assert set(ind.columns) == {
        "outer_weight",
        "normalized_weight",
        "performance",
        "scale_min",
        "scale_max",
    }
    assert list(ind.index.names) == ["lv", "indicator"]
    # normalized weights per LV must sum to 1 (within float tolerance)
    sums = ind.groupby(level="lv")["normalized_weight"].sum()
    for lv, s in sums.items():
        assert math.isclose(float(s), 1.0, abs_tol=1e-9), f"{lv}: normalized weights sum to {s}"


def test_ipma_loy_as_target_includes_sat():
    fit, _ = _satisfaction_plspm()
    res = fit.ipma("LOY").latent_variables()
    # SAT → LOY is direct; everything else flows through SAT.
    assert "SAT" in res.index
    assert "IMAG" in res.index
    # SAT direct effect should be one of the largest contributors
    sat_imp = float(res.loc["SAT", "importance"])
    assert sat_imp > 0.0
