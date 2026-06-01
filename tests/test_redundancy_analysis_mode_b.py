"""Redundancy-analysis pattern: a Mode B (formative) LV with several
correlated indicators predicts a single-indicator global rating LV.

Used in convergent-validity assessment of formative constructs (Hair et al.,
A Primer on PLS-SEM, Ch. 3). The structural path should recover the true
driver→rating relationship; weights should sum to roughly unity (post-
normalisation) and concentrate on the indicators that carry the strongest
signal.

This test generates synthetic data with a known driver–rating relationship
and asserts that the engine recovers the structural path within sampling
tolerance across multiple seeds."""

import numpy as np
import pandas as pd
import pytest

import openpls.config as c
from openpls.mode import Mode
from openpls.plspm import Plspm
from openpls.scheme import Scheme


def _redundancy_dataset(
    n: int = 500,
    seed: int = 0,
    true_path: float = 0.7,
    indicator_loads: tuple[float, ...] = (0.8, 0.7, 0.6, 0.7, 0.8),
) -> pd.DataFrame:
    """Generate (formative driver block, global rating) with a known structural path.

    The latent driver is a standard normal. Each formative indicator carries
    a fraction `indicator_loads[i]` of the latent and adds independent noise.
    The global rating loads on the latent with weight `true_path` plus noise.
    """
    rng = np.random.default_rng(seed)
    driver_latent = rng.standard_normal(n)
    indicators = {}
    for i, load in enumerate(indicator_loads, start=1):
        noise = np.sqrt(1.0 - load**2) * rng.standard_normal(n)
        indicators[f"drv_{i}"] = load * driver_latent + noise
    noise_rating = np.sqrt(1.0 - true_path**2) * rng.standard_normal(n)
    indicators["rating"] = true_path * driver_latent + noise_rating
    return pd.DataFrame(indicators)


def _fit_redundancy(data: pd.DataFrame, scheme: Scheme = Scheme.PATH) -> Plspm:
    structure = c.Structure()
    structure.add_path(["DRIVER"], ["RATING"])
    config = c.Config(structure.path(), scaled=True)
    drv_cols = [col for col in data.columns if col.startswith("drv_")]
    config.add_lv("DRIVER", Mode.B, *[c.MV(col) for col in drv_cols])
    config.add_lv("RATING", Mode.A, c.MV("rating"))
    return Plspm(data, config, scheme)


def test_redundancy_path_is_substantial_and_positive_across_seeds():
    # The Mode B composite cannot perfectly reconstruct the latent driver
    # (indicator loadings < 1), so the recovered path is the regression of
    # the rating on the empirical composite — bounded above by the true
    # latent-latent path. Assert the engine recovers a substantial positive
    # relationship rather than a tight numerical match.
    seeds = [0, 1, 2, 3, 4]
    estimates = []
    for seed in seeds:
        data = _redundancy_dataset(seed=seed)
        fit = _fit_redundancy(data)
        beta = fit.path_coefficients().loc["RATING", "DRIVER"]
        estimates.append(beta)
    mean_estimate = float(np.mean(estimates))
    assert 0.55 < mean_estimate < 0.80, (
        f"mean path estimate {mean_estimate:.3f} outside 0.55..0.80 sampling band"
    )
    assert min(estimates) > 0.40, f"individual estimate below 0.40: {min(estimates):.3f}"


def test_redundancy_r_squared_is_in_expected_band():
    data = _redundancy_dataset(seed=42)
    fit = _fit_redundancy(data)
    r2 = fit.inner_summary().loc["RATING", "r_squared"]
    # Population-bound for R² is true_path² = 0.49; sample composite
    # attenuates this. Expect 0.30..0.50 at n=500.
    assert 0.30 < r2 < 0.50, f"R²={r2:.3f} outside expected band 0.30..0.50"


def test_redundancy_global_rating_loading_is_unity():
    data = _redundancy_dataset(seed=7)
    fit = _fit_redundancy(data)
    outer = fit.outer_model()
    rating_loading = outer.loc["rating", "loading"]
    assert rating_loading == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("scheme", [Scheme.PATH, Scheme.CENTROID, Scheme.FACTORIAL, Scheme.PCA])
def test_redundancy_path_invariant_across_inner_schemes(scheme):
    """The two-LV redundancy model has a degenerate inner scheme — all inner
    schemes must produce the same path coefficient and the same outer-model
    weights."""
    data = _redundancy_dataset(seed=11)
    ref = _fit_redundancy(data, Scheme.PATH)
    candidate = _fit_redundancy(data, scheme)
    path_diff = float(
        np.max(np.abs(ref.path_coefficients().values - candidate.path_coefficients().values))
    )
    weight_diff = float(
        np.max(np.abs(ref.outer_model()["weight"].values - candidate.outer_model()["weight"].values))
    )
    assert path_diff < 1e-6, f"path differs across schemes (max |Δ|={path_diff})"
    assert weight_diff < 1e-6, f"weights differ across schemes (max |Δ|={weight_diff})"
