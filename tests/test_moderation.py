import math

import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.moderation import Moderation
from openpls.scheme import Scheme


def _satisfaction_config():
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
    return satisfaction, config


def test_moderation_default_interaction_name():
    data, config = _satisfaction_config()
    m = Moderation(data, config, predictor="IMAG", moderator="EXPE", target="SAT")
    assert m.interaction_name == "IMAG_x_EXPE"


def test_moderation_custom_interaction_name():
    data, config = _satisfaction_config()
    m = Moderation(
        data,
        config,
        predictor="IMAG",
        moderator="EXPE",
        target="SAT",
        interaction_name="MOD",
    )
    assert m.interaction_name == "MOD"
    # refit's path matrix must include the new LV with a 1 in (target, interaction)
    refit_path = m.refit().path_coefficients()
    assert "MOD" in refit_path.index or "MOD" in refit_path.columns
    inner = m.refit().inner_model()
    assert "MOD -> SAT" in inner.index


def test_moderation_refit_path_includes_interaction_to_target():
    data, config = _satisfaction_config()
    m = Moderation(data, config, predictor="IMAG", moderator="EXPE", target="SAT")
    path = m.refit().path_coefficients()
    assert "IMAG_x_EXPE" in path.columns
    # interaction → SAT must have a non-zero estimate (well-identified problem)
    val = float(path.loc["SAT", "IMAG_x_EXPE"])
    assert math.isfinite(val)
    assert val != 0.0


def test_moderation_interaction_effect_returns_expected_fields():
    data, config = _satisfaction_config()
    m = Moderation(data, config, predictor="IMAG", moderator="EXPE", target="SAT")
    eff = m.interaction_effect()
    assert set(eff.index) == {"estimate", "std error", "t", "p>|t|"}
    assert math.isfinite(float(eff["estimate"]))
    assert math.isfinite(float(eff["std error"]))


def test_moderation_base_and_refit_are_separate_fits():
    data, config = _satisfaction_config()
    m = Moderation(data, config, predictor="IMAG", moderator="EXPE", target="SAT")
    base_path = m.base().path_coefficients()
    refit_path = m.refit().path_coefficients()
    # base does not have the interaction column
    assert "IMAG_x_EXPE" not in base_path.columns
    # base path matrix dimensions: 6x6 (original LVs). refit: 7x7
    assert base_path.shape == (6, 6)
    assert refit_path.shape == (7, 7)


def test_moderation_rejects_unknown_predictor():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="predictor"):
        Moderation(data, config, predictor="NOPE", moderator="EXPE", target="SAT")


def test_moderation_rejects_unknown_moderator():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="moderator"):
        Moderation(data, config, predictor="IMAG", moderator="NOPE", target="SAT")


def test_moderation_rejects_unknown_target():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="target"):
        Moderation(data, config, predictor="IMAG", moderator="EXPE", target="NOPE")


def test_moderation_rejects_same_predictor_and_moderator():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="different"):
        Moderation(data, config, predictor="IMAG", moderator="IMAG", target="SAT")


def test_moderation_rejects_target_equal_predictor():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="cannot be"):
        Moderation(data, config, predictor="SAT", moderator="EXPE", target="SAT")


def test_moderation_rejects_exogenous_target():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="exogenous"):
        Moderation(data, config, predictor="EXPE", moderator="QUAL", target="IMAG")


def test_moderation_rejects_interaction_name_clash():
    data, config = _satisfaction_config()
    with pytest.raises(ValueError, match="clashes with an existing LV"):
        Moderation(
            data,
            config,
            predictor="IMAG",
            moderator="EXPE",
            target="SAT",
            interaction_name="QUAL",
        )


def test_moderation_uses_path_scheme():
    data, config = _satisfaction_config()
    m = Moderation(
        data,
        config,
        predictor="IMAG",
        moderator="EXPE",
        target="SAT",
        scheme=Scheme.PATH,
    )
    # just verify it ran and produced an interaction estimate
    val = float(m.refit().path_coefficients().loc["SAT", "IMAG_x_EXPE"])
    assert math.isfinite(val)
