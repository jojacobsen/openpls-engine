import math

import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def satisfaction_path_matrix():
    structure = c.Structure()
    structure.add_path(["IMAG"], ["EXPE", "SAT", "LOY"])
    structure.add_path(["EXPE"], ["QUAL", "VAL", "SAT"])
    structure.add_path(["QUAL"], ["VAL", "SAT"])
    structure.add_path(["VAL"], ["SAT"])
    structure.add_path(["SAT"], ["LOY"])
    return structure.path()


def _satisfaction_plspm():
    satisfaction = pd.read_csv("file:tests/data/satisfaction.csv", index_col=0)
    config = c.Config(satisfaction_path_matrix(), scaled=False)
    for lv in ["IMAG", "EXPE", "QUAL", "VAL", "SAT", "LOY"]:
        config.add_lv_with_columns_named(lv, Mode.A, satisfaction, lv.lower())
    return Plspm(satisfaction, config, Scheme.CENTROID)


def test_model_fit_srmr_in_reasonable_range():
    plspm_calc = _satisfaction_plspm()
    fit = plspm_calc.model_fit()
    srmr = fit.srmr()
    assert math.isfinite(srmr)
    # ECSI / satisfaction is the canonical mode-A example. SRMR should be
    # comfortably below the conventional 0.10 acceptable-fit cutoff.
    assert 0.0 <= srmr <= 0.10, f"SRMR={srmr} outside expected range"


def test_model_fit_duls_nonnegative():
    fit = _satisfaction_plspm().model_fit()
    d_uls = fit.d_uls()
    assert math.isfinite(d_uls)
    assert d_uls >= 0.0


def test_model_fit_residuals_shape():
    fit = _satisfaction_plspm().model_fit()
    resid = fit.residuals()
    # 24 indicators total: 5 for each of IMAG, EXPE, QUAL + 3 for VAL + 3 for SAT + 4 for LOY
    assert resid.shape[0] == resid.shape[1]
    assert resid.shape[0] >= 20
    # Residual matrix is symmetric.
    pd_diff = (resid - resid.T).abs().to_numpy().max()
    assert pd_diff < 1e-10


def test_model_fit_summary_has_both_metrics():
    summary = _satisfaction_plspm().model_fit().summary()
    assert list(summary.columns) == ["srmr", "d_uls"]
    assert summary.shape == (1, 2)


def test_mode_b_within_lv_pairs_excluded_from_fit():
    """Regression: saturated-model fit (SRMR / d_ULS) must skip indicator pairs
    that belong to the same Mode B (formative) LV. Their inter-correlation is
    empirical, not implied by Λ Φ Λᵀ, so including them inflates the residual
    sums purely as a measurement-model artifact (Henseler et al. 2014 §5.3).

    Construction: two LVs, one Mode A driver (DRV) with a strong common factor
    and one Mode B composite (CMP) whose indicators are intentionally weakly
    inter-correlated. Without the Mode B exemption, the within-CMP pairs
    dominate d_ULS / SRMR; with the exemption, d_ULS comes down to the
    cross-LV + within-DRV residuals only.
    """
    import numpy as np

    rng = np.random.default_rng(11)
    n = 400
    drv_factor = rng.standard_normal(n)
    drv_frame = {
        f"d{i}": drv_factor + rng.standard_normal(n) * 0.3 for i in range(1, 5)
    }
    cmp_frame = {f"c{i}": rng.standard_normal(n) for i in range(1, 4)}
    cmp_frame["c1"] = cmp_frame["c1"] + 0.4 * drv_factor
    cmp_frame["c2"] = cmp_frame["c2"] + 0.4 * drv_factor
    cmp_frame["c3"] = cmp_frame["c3"] + 0.4 * drv_factor
    data = pd.DataFrame({**drv_frame, **cmp_frame})

    structure = c.Structure()
    structure.add_path(["DRV"], ["CMP"])
    config = c.Config(structure.path())
    config.add_lv_with_columns_named("DRV", Mode.A, data, "d")
    config.add_lv_with_columns_named("CMP", Mode.B, data, "c")

    fit = Plspm(data, config, Scheme.PATH).model_fit()

    n_ind = 7
    n_within_b = 3
    n_within_b_pairs = n_within_b * (n_within_b - 1) // 2
    n_pairs_total = n_ind * (n_ind - 1) // 2
    n_pairs_kept = n_pairs_total - n_within_b_pairs

    resid = fit.residuals().to_numpy()
    cmp_idx = [list(data.columns).index(f"c{i}") for i in (1, 2, 3)]
    full_lower = resid[np.tri(n_ind, n_ind, k=-1, dtype=bool)]
    naive_d_uls = float(np.sum(full_lower ** 2))
    within_cmp_resid = []
    for a in cmp_idx:
        for b in cmp_idx:
            if a > b:
                within_cmp_resid.append(resid[a, b])
    expected_d_uls = naive_d_uls - float(np.sum(np.asarray(within_cmp_resid) ** 2))
    expected_srmr = float(math.sqrt(expected_d_uls / n_pairs_kept))

    assert math.isclose(fit.d_uls(), expected_d_uls, rel_tol=1e-9), (
        f"d_ULS={fit.d_uls()} vs expected (within-CMP excluded)={expected_d_uls}"
    )
    assert math.isclose(fit.srmr(), expected_srmr, rel_tol=1e-9), (
        f"SRMR={fit.srmr()} vs expected (within-CMP excluded)={expected_srmr}"
    )
