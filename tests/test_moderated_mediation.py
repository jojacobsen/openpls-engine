import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.moderated_mediation import ModeratedMediation
from openpls.scheme import Scheme


def _satisfaction_config():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    config = c.Config(structure.path(), scaled=True)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return satisfaction, config


def _first_stage():
    # EXPE moderates QUAL -> VAL in QUAL -> VAL -> SAT.
    data, config = _satisfaction_config()
    return ModeratedMediation(
        data, config, "QUAL", "VAL", "SAT", "EXPE", stage="first", scheme=Scheme.PATH
    )


def _second_stage():
    # IMAG moderates SAT -> LOY in VAL -> SAT -> LOY.
    data, config = _satisfaction_config()
    return ModeratedMediation(
        data, config, "VAL", "SAT", "LOY", "IMAG", stage="second", scheme=Scheme.PATH
    )


# Reference values: seminr 2.6.0, two_stage interaction, path weighting,
# same satisfaction model (all four loy indicators), a3/b3 and the index
# computed from its path coefficients.
def test_first_stage_matches_seminr():
    mm = _first_stage()
    coef = mm.coefficients()
    assert coef["a"] == pytest.approx(0.644637, abs=1e-5)
    assert coef["interaction"] == pytest.approx(-0.054178, abs=1e-5)
    assert coef["b"] == pytest.approx(0.580416, abs=1e-5)
    assert mm.index() == pytest.approx(-0.031446, abs=1e-5)


def test_second_stage_matches_seminr():
    mm = _second_stage()
    coef = mm.coefficients()
    assert coef["a"] == pytest.approx(0.580059, abs=1e-5)
    assert coef["b"] == pytest.approx(0.489158, abs=1e-5)
    assert coef["interaction"] == pytest.approx(0.030523, abs=1e-5)
    assert mm.index() == pytest.approx(0.017705, abs=1e-5)


def test_index_is_the_slope_of_the_conditional_indirect_effect():
    for mm in (_first_stage(), _second_stage()):
        cond = mm.conditional_indirect_effects(levels=(-1.0, 0.0, 1.0, 2.0))["estimate"]
        assert cond[1.0] - cond[0.0] == pytest.approx(mm.index())
        assert cond[0.0] - cond[-1.0] == pytest.approx(mm.index())
        assert cond[2.0] - cond[1.0] == pytest.approx(mm.index())
        coef = mm.coefficients()
        # At the moderator mean the indirect effect is the plain product a·b.
        assert cond[0.0] == pytest.approx(coef["a"] * coef["b"])


def test_coefficients_come_from_the_fit_with_the_interaction():
    mm = _first_stage()
    pc = mm.moderation().refit().path_coefficients()
    coef = mm.coefficients()
    assert coef["a"] == pytest.approx(pc.loc["VAL", "QUAL"])
    assert coef["b"] == pytest.approx(pc.loc["SAT", "VAL"])
    assert coef["interaction (standardized)"] == pytest.approx(
        pc.loc["VAL", mm.interaction_name]
    )
    assert mm.interaction_name == "QUAL_x_EXPE"
    assert _second_stage().interaction_name == "SAT_x_IMAG"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"stage": "both"}, "stage must be one of"),
        ({"moderator": "QUAL"}, "must be different LVs"),
        ({"moderator": "NOPE"}, "is not a latent variable"),
        # IMAG has no path into VAL: the moderator main effect is missing.
        ({"moderator": "IMAG"}, "add the main effect IMAG -> VAL"),
        # LOY is not reached from VAL directly.
        ({"outcome": "LOY"}, "needs the path VAL -> LOY"),
    ],
)
def test_invalid_specifications_raise(kwargs, message):
    data, config = _satisfaction_config()
    spec = {
        "predictor": "QUAL",
        "mediator": "VAL",
        "outcome": "SAT",
        "moderator": "EXPE",
        "stage": "first",
    }
    spec.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ModeratedMediation(data, config, **spec)


def test_bootstrap_shape_and_determinism():
    mm = _first_stage()
    boot = mm.bootstrap(n_boot=20, seed=3)
    idx = boot.index()
    assert list(idx.index) == [
        "original", "mean", "std.error", "perc.lower", "perc.upper", "t stat."
    ]
    assert idx["original"] == pytest.approx(mm.index())
    assert idx["perc.lower"] <= idx["perc.upper"]
    assert boot.n_valid + boot.n_failed == 20
    cond = boot.conditional_indirect_effects()
    assert list(cond.index) == [-1.0, 0.0, 1.0]
    assert (cond["perc.lower"] <= cond["perc.upper"]).all()

    again = mm.bootstrap(n_boot=20, seed=3)
    pd.testing.assert_series_equal(boot.index(), again.index())


def test_bootstrap_does_not_depend_on_n_jobs():
    mm = _first_stage()
    serial = mm.bootstrap(n_boot=8, seed=11, n_jobs=1)
    parallel = mm.bootstrap(n_boot=8, seed=11, n_jobs=2)
    pd.testing.assert_series_equal(serial.index(), parallel.index())
    pd.testing.assert_frame_equal(
        serial.conditional_indirect_effects(), parallel.conditional_indirect_effects()
    )


@pytest.mark.parametrize("kwargs", [{"n_boot": 1}, {"alpha": 1.5}, {"n_jobs": 0}])
def test_bootstrap_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        _first_stage().bootstrap(**kwargs)


def _synthetic(interaction: float, n: int = 400, seed: int = 0):
    """X -> M -> Y with W -> M and an X·W effect on M, three indicators each."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    w = rng.standard_normal(n)
    m = 0.5 * x + 0.3 * w + interaction * x * w + 0.6 * rng.standard_normal(n)
    y = 0.6 * m + 0.6 * rng.standard_normal(n)
    cols = {}
    for name, lv in (("x", x), ("w", w), ("m", m), ("y", y)):
        for i in range(1, 4):
            cols[f"{name}{i}"] = lv + 0.5 * rng.standard_normal(n)
    data = pd.DataFrame(cols)
    structure = c.Structure()
    structure.add_path(["X", "W"], ["M"])
    structure.add_path(["M"], ["Y"])
    config = c.Config(structure.path(), scaled=True)
    for lv in ["X", "W", "M", "Y"]:
        config.add_lv_with_columns_named(lv, Mode.A, data, lv.lower())
    return data, config


def test_bootstrap_detects_a_true_moderated_mediation():
    data, config = _synthetic(interaction=0.4)
    mm = ModeratedMediation(data, config, "X", "M", "Y", "W", stage="first")
    assert mm.index() > 0
    boot = mm.bootstrap(n_boot=60, seed=1, n_jobs=2)
    assert boot.significant()
    cond = boot.conditional_indirect_effects()["original"]
    assert cond[-1.0] < cond[0.0] < cond[1.0]


def test_index_is_small_without_moderation():
    # A 95% CI excludes zero in 5% of null datasets by design, so assert the
    # index is within three bootstrap SEs of zero instead. (On this DGP the
    # bootstrap SE matches the Monte Carlo SD of the index: 0.0221 vs 0.0217
    # over 400 datasets.)
    data, config = _synthetic(interaction=0.0)
    mm = ModeratedMediation(data, config, "X", "M", "Y", "W", stage="first")
    boot = mm.bootstrap(n_boot=60, seed=1, n_jobs=2)
    assert abs(boot.index()["t stat."]) < 3
