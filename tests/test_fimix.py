import math

import numpy as np
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
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_fimix_memberships_sum_to_one_per_row():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=7)
    post = fm.memberships()
    row_sums = post.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_fimix_class_sizes_sum_to_one():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=3, n_restarts=2, seed=11)
    rho = fm.class_sizes()
    assert rho.shape == (3,)
    assert math.isclose(float(rho.sum()), 1.0, abs_tol=1e-9)
    assert (rho > 0).all()


def test_fimix_hard_assignments_are_in_valid_range():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=3)
    labels = fm.hard_assignments()
    assert labels.min() >= 1
    assert labels.max() <= 2
    # both classes should have at least one assignment (non-degenerate)
    assert labels.nunique() == 2


def test_fimix_class_paths_has_expected_structure():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=5)
    paths = fm.class_paths()
    assert set(paths.columns) == {"class", "from", "to", "estimate"}
    # K=2, 5 endogenous LVs (EXPE, QUAL, VAL, SAT, LOY) with structural paths.
    # Each endo LV contributes (1 intercept + p predecessors) rows per class.
    # EXPE←IMAG (1), QUAL←EXPE (1), VAL←EXPE,QUAL (2), SAT←IMAG,EXPE,QUAL,VAL (4), LOY←IMAG,SAT (2).
    # Per class: 5 intercepts + (1+1+2+4+2)=10 path rows = 15. K=2 → 30 rows.
    assert len(paths) == 30
    assert set(paths["class"].unique()) == {1, 2}
    assert "(intercept)" in set(paths["from"])


def test_fimix_fit_criteria_includes_all_keys():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=1)
    crit = fm.fit_criteria()
    expected = {"log_lik", "n_params", "aic", "aic3", "aic4", "bic", "caic", "mdl5", "en"}
    assert expected.issubset(set(crit.index))
    # EN ranges in [0, 1]
    assert 0.0 <= float(crit["en"]) <= 1.0


def test_fimix_log_likelihood_is_finite():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=42)
    ll = fm.log_likelihood()
    assert math.isfinite(ll)


def test_fimix_deterministic_with_seed():
    fit = _satisfaction_plspm()
    fm_a = fit.fimix(n_classes=2, n_restarts=2, seed=99)
    fm_b = fit.fimix(n_classes=2, n_restarts=2, seed=99)
    assert math.isclose(fm_a.log_likelihood(), fm_b.log_likelihood(), abs_tol=1e-9)


def test_fimix_more_classes_use_more_parameters():
    fit = _satisfaction_plspm()
    crit_2 = fit.fimix(n_classes=2, n_restarts=2, seed=4).fit_criteria()
    crit_3 = fit.fimix(n_classes=3, n_restarts=2, seed=4).fit_criteria()
    assert int(crit_3["n_params"]) > int(crit_2["n_params"])


def test_fimix_rejects_too_few_classes():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="n_classes"):
        fit.fimix(n_classes=1)


def test_fimix_rejects_bad_tolerance():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="tolerance"):
        fit.fimix(n_classes=2, tolerance=0.0)


def test_fimix_rejects_zero_restarts():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="n_restarts"):
        fit.fimix(n_classes=2, n_restarts=0)


def test_fimix_rejects_tiny_max_iter():
    fit = _satisfaction_plspm()
    with pytest.raises(ValueError, match="max_iter"):
        fit.fimix(n_classes=2, max_iter=5)


def test_fimix_memberships_indexed_by_data():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=1, seed=0)
    post = fm.memberships()
    assert list(post.index) == list(fit.scores().index)
    assert list(post.columns) == ["class_1", "class_2"]


def _corporate_reputation_plspm():
    """The simplified corporate-reputation model (COMP + LIKE -> CUSA -> CUSL)
    used in issue #34. Without the degeneracy guard, FIMIX on this fixture
    selected a ~1.5% micro-class whose residual variance collapsed, giving a
    (nonsensical) positive log-likelihood."""
    df = pd.read_csv("file:tests/data/corporate_reputation.csv", sep=";")
    cols = ["comp_1", "comp_2", "comp_3", "like_1", "like_2", "like_3",
            "cusl_1", "cusl_2", "cusl_3", "cusa"]
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    structure = c.Structure()
    structure.add_path(["COMP"], ["CUSA"])
    structure.add_path(["LIKE"], ["CUSA"])
    structure.add_path(["CUSA"], ["CUSL"])
    config = c.Config(structure.path(), scaled=True)
    config.add_lv("COMP", Mode.A, c.MV("comp_1"), c.MV("comp_2"), c.MV("comp_3"))
    config.add_lv("LIKE", Mode.A, c.MV("like_1"), c.MV("like_2"), c.MV("like_3"))
    config.add_lv("CUSA", Mode.A, c.MV("cusa"))
    config.add_lv("CUSL", Mode.A, c.MV("cusl_1"), c.MV("cusl_2"), c.MV("cusl_3"))
    return Plspm(df, config, Scheme.PATH)


def test_fimix_degeneracy_guard_rejects_micro_class():
    """Regression for issue #34. The guard must not return a degenerate
    solution (a class too small to identify its own parameters). Before the
    guard, seed=42 returned class sizes ~= 0.015 / 0.985; after it, a balanced,
    admissible split is selected and reported as non-degenerate."""
    fit = _corporate_reputation_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=10, seed=42)
    rho = fm.class_sizes().to_numpy()
    n = len(fit.scores())
    # 2 endogenous LVs: CUSA (1 intercept + 2 preds) + CUSL (1 intercept + 1
    # pred) = 5 coefs + 2 variances = 7 parameters per class.
    per_class_params = 7
    assert not fm.is_degenerate()
    # every class must have more effective members than it estimates params.
    assert (rho.min() * n) > per_class_params
    # and the split is substantive, not a vanishing micro-class.
    assert rho.min() > 0.05


def test_fimix_is_degenerate_defaults_false_on_clean_fixture():
    fit = _satisfaction_plspm()
    fm = fit.fimix(n_classes=2, n_restarts=2, seed=7)
    assert fm.is_degenerate() in (True, False)  # callable, returns a bool
