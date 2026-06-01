"""Structural-path recovery on a synthetic three-LV mediation chain.

Generates X (3 reflective indicators) → M (3 reflective indicators) →
Y (3 reflective indicators) with known structural coefficients, and asserts
that the engine recovers the population paths within sampling tolerance
across several seeds. Also exercises an indirect-effect bookkeeping path.

The intent is a self-contained correctness check that does not rely on
any external reference: ground truth is fixed at data-generation time.
"""

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


TRUE_X_TO_M = 0.6
TRUE_M_TO_Y = 0.55
TRUE_X_TO_Y_DIRECT = 0.25
TRUE_INDIRECT = TRUE_X_TO_M * TRUE_M_TO_Y
TRUE_TOTAL_X_TO_Y = TRUE_X_TO_Y_DIRECT + TRUE_INDIRECT


def _three_lv_mediation(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_latent = rng.standard_normal(n)
    m_latent = TRUE_X_TO_M * x_latent + np.sqrt(1.0 - TRUE_X_TO_M**2) * rng.standard_normal(n)
    y_var = (
        TRUE_X_TO_Y_DIRECT**2
        + TRUE_M_TO_Y**2
        + 2.0 * TRUE_X_TO_Y_DIRECT * TRUE_M_TO_Y * TRUE_X_TO_M
    )
    y_noise_sd = np.sqrt(max(1.0 - y_var, 1e-6))
    y_latent = (
        TRUE_X_TO_Y_DIRECT * x_latent
        + TRUE_M_TO_Y * m_latent
        + y_noise_sd * rng.standard_normal(n)
    )

    def reflective_block(latent: np.ndarray, prefix: str, loadings=(0.85, 0.78, 0.82)):
        cols = {}
        for i, load in enumerate(loadings, start=1):
            noise = np.sqrt(1.0 - load**2) * rng.standard_normal(n)
            cols[f"{prefix}_{i}"] = load * latent + noise
        return cols

    block = {}
    block.update(reflective_block(x_latent, "x"))
    block.update(reflective_block(m_latent, "m"))
    block.update(reflective_block(y_latent, "y"))
    return pd.DataFrame(block)


def _fit_mediation(data: pd.DataFrame, scheme: Scheme = Scheme.PATH) -> Plspm:
    structure = c.Structure()
    structure.add_path(["X"], ["M", "Y"])
    structure.add_path(["M"], ["Y"])
    config = c.Config(structure.path(), scaled=True)
    config.add_lv_with_columns_named("X", Mode.A, data, "x")
    config.add_lv_with_columns_named("M", Mode.A, data, "m")
    config.add_lv_with_columns_named("Y", Mode.A, data, "y")
    return Plspm(data, config, scheme)


def test_direct_paths_recovered_across_seeds():
    # path_coefficients() is indexed [target_row, source_col].
    estimates_x_m = []
    estimates_m_y = []
    estimates_x_y = []
    for seed in range(5):
        data = _three_lv_mediation(seed=seed)
        fit = _fit_mediation(data)
        paths = fit.path_coefficients()
        estimates_x_m.append(paths.loc["M", "X"])
        estimates_m_y.append(paths.loc["Y", "M"])
        estimates_x_y.append(paths.loc["Y", "X"])
    # Reflective indicators with loadings around 0.8 attenuate the recovered
    # path slightly. Use a 0.10 tolerance band on the cross-seed mean.
    assert abs(float(np.mean(estimates_x_m)) - TRUE_X_TO_M) < 0.10
    assert abs(float(np.mean(estimates_m_y)) - TRUE_M_TO_Y) < 0.10
    assert abs(float(np.mean(estimates_x_y)) - TRUE_X_TO_Y_DIRECT) < 0.10


def test_indirect_effect_via_mediator():
    data = _three_lv_mediation(seed=99)
    fit = _fit_mediation(data)
    effects = fit.effects()
    indirect_row = effects[(effects["from"] == "X") & (effects["to"] == "Y")]
    assert not indirect_row.empty
    indirect = float(indirect_row["indirect"].iloc[0])
    direct = float(indirect_row["direct"].iloc[0])
    total = float(indirect_row["total"].iloc[0])
    # Wider tolerance because reflective-indicator attenuation compounds
    # along the X→M→Y chain.
    assert abs(indirect - TRUE_INDIRECT) < 0.15
    assert abs(direct - TRUE_X_TO_Y_DIRECT) < 0.15
    assert abs(total - TRUE_TOTAL_X_TO_Y) < 0.15


def test_r_squared_on_y_in_expected_band():
    data = _three_lv_mediation(seed=12)
    fit = _fit_mediation(data)
    r2 = fit.inner_summary().loc["Y", "r_squared"]
    # Population R² on Y under unit-variance latents:
    # TRUE_X_TO_Y_DIRECT² + TRUE_M_TO_Y² + 2*direct*med*β_xm. Reflective
    # measurement with loadings ~0.8 attenuates the empirical R² noticeably.
    pop_r2 = (
        TRUE_X_TO_Y_DIRECT**2
        + TRUE_M_TO_Y**2
        + 2.0 * TRUE_X_TO_Y_DIRECT * TRUE_M_TO_Y * TRUE_X_TO_M
    )
    assert abs(r2 - pop_r2) < 0.20
