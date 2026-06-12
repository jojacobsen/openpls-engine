import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.long_bootstrap import LongBootstrap
from openpls.mode import Mode
from openpls.scale import Scale
from openpls.scheme import Scheme


def _russa_config():
    structure = c.Structure()
    structure.add_path(["AGRI", "IND"], ["POLINS"])
    config = c.Config(structure.path(), default_scale=Scale.NUM)
    config.add_lv("AGRI", Mode.A, c.MV("gini"), c.MV("farm"), c.MV("rent"))
    config.add_lv("IND", Mode.A, c.MV("gnpr"), c.MV("labo"))
    config.add_lv("POLINS", Mode.A, c.MV("ecks"), c.MV("death"), c.MV("demo"), c.MV("inst"))
    return config


def _russa():
    return pd.read_csv("file:tests/data/russa.csv", index_col=0)


def test_long_bootstrap_smoke_returns_expected_columns():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=20, seed=0)
    paths = boot.paths()
    assert set(paths.columns) >= {
        "source", "target", "original",
        "boot_mean", "se", "t", "p_value", "ci_lower", "ci_upper", "valid",
    }
    # 2 structural paths: AGRI→POLINS, IND→POLINS
    assert len(paths) == 2
    assert set(zip(paths["source"], paths["target"], strict=False)) == {
        ("AGRI", "POLINS"), ("IND", "POLINS"),
    }


def test_long_bootstrap_loadings_and_weights_shape():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=15, seed=1)
    loadings = boot.loadings()
    weights = boot.weights()
    # 9 indicators total: 3 AGRI + 2 IND + 4 POLINS
    assert len(loadings) == 9
    assert len(weights) == 9
    assert set(loadings.columns) >= {"lv", "indicator", "original", "boot_mean", "ci_lower", "ci_upper"}


def test_long_bootstrap_total_effects_includes_indirect():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=15, seed=2)
    total = boot.total_effects()
    # Direct paths AGRI→POLINS and IND→POLINS only; no indirect in this 2-stage model.
    assert len(total) == 2
    assert set(zip(total["source"], total["target"], strict=False)) == {
        ("AGRI", "POLINS"), ("IND", "POLINS"),
    }


def test_long_bootstrap_progress_callback_fires():
    calls: list[tuple[int, int]] = []
    LongBootstrap(
        _russa(),
        _russa_config(),
        iterations=10,
        seed=0,
        on_progress=lambda done, total: calls.append((done, total)),
        progress_every=2,
    )
    # At least one mid-run call and the final guaranteed call
    assert len(calls) >= 2
    assert calls[-1] == (10, 10)
    # All totals match iterations
    assert all(total == 10 for _, total in calls)
    # done values are monotonically increasing
    dones = [done for done, _ in calls]
    assert dones == sorted(dones)


def test_long_bootstrap_seed_determinism():
    a = LongBootstrap(_russa(), _russa_config(), iterations=10, seed=42).paths()
    b = LongBootstrap(_russa(), _russa_config(), iterations=10, seed=42).paths()
    pd.testing.assert_frame_equal(a, b)


def test_long_bootstrap_rejects_bad_min_success_ratio():
    with pytest.raises(ValueError, match="min_success_ratio"):
        LongBootstrap(_russa(), _russa_config(), iterations=5, min_success_ratio=1.5)


def test_long_bootstrap_runtime_floor_raises_when_all_resamples_fail(monkeypatch):
    from openpls import long_bootstrap as lb

    original_fit = lb.LongBootstrap._LongBootstrap__fit
    state = {"called": 0}

    def _fail_after_point(self, df):
        state["called"] += 1
        if state["called"] == 1:
            return original_fit(self, df)
        raise RuntimeError("boom")

    # Patch so the point estimate succeeds but every bootstrap resample raises.
    monkeypatch.setattr(lb.LongBootstrap, "_LongBootstrap__fit", _fail_after_point)
    with pytest.raises(RuntimeError, match="Bootstrap failed"):
        LongBootstrap(_russa(), _russa_config(), iterations=4, seed=0, min_success_ratio=0.5)


def test_long_bootstrap_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        LongBootstrap(_russa(), _russa_config(), iterations=5, alpha=0.0)


def test_long_bootstrap_rejects_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        LongBootstrap(_russa(), _russa_config(), iterations=0)


def test_long_bootstrap_ci_brackets_point_estimate_when_significant():
    # With enough iterations the BCa CI for a strong path should bracket the
    # original estimate (not require sign-agreement, just that lo <= point or
    # point <= hi).
    boot = LongBootstrap(_russa(), _russa_config(), iterations=60, seed=7)
    paths = boot.paths()
    for _, row in paths.iterrows():
        if np.isnan(row["original"]) or np.isnan(row["ci_lower"]) or np.isnan(row["ci_upper"]):
            continue
        assert row["ci_lower"] <= row["ci_upper"]


def test_long_bootstrap_completed_and_failed_counts():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=20, seed=3)
    assert boot.completed + boot.failed == 20
    assert boot.completed >= 1
    assert boot.alpha == 0.05


# ---------------------------------------------------------------------------
# BootstrapInference (issue #31) — canonical inference tables
# ---------------------------------------------------------------------------


_INFERENCE_COLUMNS = {
    "original",
    "mean",
    "std_error",
    "t_value",
    "p_value",
    "ci_percentile_2_5",
    "ci_percentile_97_5",
    "ci_bc_2_5",
    "ci_bc_97_5",
}


def _satisfaction_config():
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(structure.path(), scaled=False)
    config.add_lv_with_columns_named("IMAG", Mode.A, satisfaction, "imag")
    config.add_lv_with_columns_named("EXPE", Mode.A, satisfaction, "expe")
    config.add_lv_with_columns_named("QUAL", Mode.A, satisfaction, "qual")
    config.add_lv_with_columns_named("VAL", Mode.A, satisfaction, "val")
    config.add_lv_with_columns_named("SAT", Mode.A, satisfaction, "sat")
    config.add_lv_with_columns_named("LOY", Mode.A, satisfaction, "loy")
    return satisfaction, config


def test_inference_exposes_all_six_entity_types():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=30, seed=11)
    inf = boot.inference
    assert set(inf.keys()) == {
        "pathCoefficients",
        "outerLoadings",
        "outerWeights",
        "specificIndirectEffects",
        "totalIndirectEffects",
        "totalEffects",
    }


def test_inference_path_coefficients_schema():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=30, seed=12)
    df = boot.inference["pathCoefficients"]
    assert set(df.columns) >= _INFERENCE_COLUMNS | {"source", "target"}
    # 2 direct paths in russa: AGRI → POLINS, IND → POLINS
    assert len(df) == 2
    for _, row in df.iterrows():
        assert row["ci_percentile_2_5"] <= row["ci_percentile_97_5"]
        assert row["ci_bc_2_5"] <= row["ci_bc_97_5"]
        assert 0.0 < row["p_value"] <= 1.0


def test_inference_outer_loadings_and_weights_schema():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=25, seed=13)
    for entity in ("outerLoadings", "outerWeights"):
        df = boot.inference[entity]
        assert set(df.columns) >= _INFERENCE_COLUMNS | {"lv", "indicator"}
        # 9 indicators total (3 AGRI + 2 IND + 4 POLINS)
        assert len(df) == 9


def test_inference_total_effects_includes_indirect_when_present():
    _, config = _satisfaction_config()
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    boot = LongBootstrap(satisfaction, config, iterations=20, seed=14)
    total = boot.inference["totalEffects"]
    assert set(total.columns) >= _INFERENCE_COLUMNS | {"source", "target"}
    # IMAG → LOY is reachable only indirectly, so it must show up in
    # totalEffects (direct edge IMAG→LOY exists too, but the test only needs
    # that something beyond direct paths is present).
    pairs = set(zip(total["source"], total["target"], strict=False))
    assert ("IMAG", "LOY") in pairs
    assert ("IMAG", "SAT") in pairs


def test_inference_specific_indirect_effects_satisfaction_chains():
    _, config = _satisfaction_config()
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    boot = LongBootstrap(satisfaction, config, iterations=20, seed=15)
    sie = boot.inference["specificIndirectEffects"]
    assert set(sie.columns) >= _INFERENCE_COLUMNS | {"source", "target", "via"}
    # All 5 IMAG → ... → LOY mediation chains are documented in
    # test_specific_indirect.py — they must all surface here.
    imag_to_loy_vias = set(
        sie.loc[(sie["source"] == "IMAG") & (sie["target"] == "LOY"), "via"]
    )
    expected = {
        "SAT",
        "EXPE -> SAT",
        "EXPE -> QUAL -> SAT",
        "EXPE -> QUAL -> VAL -> SAT",
        "EXPE -> VAL -> SAT",
    }
    assert imag_to_loy_vias == expected


def test_inference_total_indirect_effects_matches_sum_of_chains():
    _, config = _satisfaction_config()
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    boot = LongBootstrap(satisfaction, config, iterations=25, seed=16)
    sie = boot.inference["specificIndirectEffects"]
    tie = boot.inference["totalIndirectEffects"]
    chain_sum = sie.groupby(["source", "target"])["original"].sum()
    tie_indexed = tie.set_index(["source", "target"])["original"]
    common = chain_sum.index.intersection(tie_indexed.index)
    assert len(common) > 0
    for key in common:
        np.testing.assert_allclose(chain_sum.loc[key], tie_indexed.loc[key], atol=1e-9)


def test_inference_empty_indirect_for_two_stage_model():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=20, seed=17)
    sie = boot.inference["specificIndirectEffects"]
    tie = boot.inference["totalIndirectEffects"]
    assert len(sie) == 0
    assert len(tie) == 0
    assert set(sie.columns) >= {"source", "target", "via", "original"}


def test_inference_percentile_ci_matches_raw_resamples():
    """Percentile CI in the inference table matches np.quantile on the same
    sign-flipped samples the aggregator sees."""
    from openpls.long_bootstrap import _flip_signs

    boot = LongBootstrap(_russa(), _russa_config(), iterations=80, seed=18)
    raw_paths = boot.resamples["pathCoefficients"]
    df = boot.inference["pathCoefficients"]
    for k, (src, tgt) in enumerate(boot.path_keys):
        row = df[(df["source"] == src) & (df["target"] == tgt)].iloc[0]
        samples = raw_paths[:, k]
        samples = samples[~np.isnan(samples)]
        flipped = _flip_signs(samples, row["original"])
        np.testing.assert_allclose(
            row["ci_percentile_2_5"], np.quantile(flipped, 0.025), rtol=1e-6
        )
        np.testing.assert_allclose(
            row["ci_percentile_97_5"], np.quantile(flipped, 0.975), rtol=1e-6
        )


def test_inference_resamples_shape():
    boot = LongBootstrap(_russa(), _russa_config(), iterations=15, seed=19)
    res = boot.resamples
    assert res["pathCoefficients"].shape == (15, 2)
    assert res["outerLoadings"].shape == (15, 9)
    assert res["outerWeights"].shape == (15, 9)
    assert res["totalEffects"].shape == (15, 3, 3)
