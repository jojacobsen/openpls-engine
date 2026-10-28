#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob / OpenPLS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Overall model-fit test via the Bollen-Stine bootstrap.

Tests whether the *estimated* (structural) model's implied indicator
correlation matrix is consistent with the empirical one, or whether the
misfit is larger than sampling fluctuation would produce if the model were
correct (Dijkstra & Henseler 2015). This is the *inferential* fit test the
Hair/SmartPLS tradition omits; it complements the descriptive SRMR.

Phase 1 covers common-factor (reflective / PLSc) models. The model-implied
matrix uses the **structural** construct-correlation reproduction

    Σ_η = (I − B)⁻¹ Ψ (I − B)⁻ᵀ ,   Σ̂ = Λ_c Φ Λ_cᵀ,  diag(Σ̂) = 1

where B is the PLSc path matrix, Ψ = diag(1 − R²) (exogenous: 1), Φ is Σ_η
standardized to a correlation, and Λ_c are the PLSc-corrected loadings. This
is NOT the saturated measurement-only matrix behind ``PLSc.srmr()``.

Three discrepancy measures are reported, matching cSEM's ``testOMF``:
``SRMR`` and ``d_ULS`` (squared Euclidean, cSEM ``dL``) over the strict lower
triangle, and the geodesic distance ``d_G`` over the full matrices. A measure
is *rejected* when its observed value exceeds the 95% quantile of the
Bollen-Stine reference distribution.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode
from openpls.plsc import InadmissiblePLScError
from openpls.scheme import Scheme

_MEASURES = ("srmr", "d_uls", "d_g")


@dataclass(frozen=True)
class ModelTestMeasure:
    """Verdict for one discrepancy measure."""

    measure: str
    observed: float
    hi95: float
    hi99: float
    p_value: float
    rejected: bool  # observed > hi95 (misfit significant at 5%)


def _matrix_sqrt(m: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Symmetric (inverse) square root via eigendecomposition.

    ``m`` must be symmetric; tiny/negative eigenvalues are floored for
    numerical robustness.
    """
    w, v = np.linalg.eigh((m + m.T) / 2.0)
    w = np.clip(w, 1e-12, None)
    d = 1.0 / np.sqrt(w) if inverse else np.sqrt(w)
    return (v * d) @ v.T


def _structural_sigma(
    plsc, config: c.Config, lvs: list[str], inds: list[str], ind_to_lv: dict[str, str]
) -> np.ndarray:
    """Structural-model-implied indicator correlation matrix Σ̂.

    Reproduces the construct correlation matrix from the estimated structural
    model block-wise (exogenous block = disattenuated empirical correlations;
    endogenous blocks reproduced from the paths), then maps it through the
    loadings. Matches cSEM's ``fit()`` to ~4 decimals:

        ζ_i   = 1 − Σ_j β_ij Φ_ij                      (disturbance variance)
        Φ_eo  = Φ_ee Γᵀ (I−B)⁻ᵀ                         (exo–endo)
        Φ_oo  = (I−B)⁻¹ (Γ Φ_ee Γᵀ + diag(ζ)) (I−B)⁻ᵀ, diag = 1
        Σ̂     = Λ Φ Λᵀ, diag(Σ̂) = 1
    """
    phi_full = plsc.adjusted_correlations().reindex(index=lvs, columns=lvs).to_numpy(dtype=float)
    # Path[i, j] = coefficient from predictor j to endogenous i.
    path = plsc.path_coefficients().reindex(index=lvs, columns=lvs).fillna(0.0).to_numpy(dtype=float)
    endo = [i for i in range(len(lvs)) if path[i].any()]
    exo = [i for i in range(len(lvs)) if not path[i].any()]

    B = path[np.ix_(endo, endo)]
    Gamma = path[np.ix_(endo, exo)]
    Phi_ee = phi_full[np.ix_(exo, exo)]
    zeta = 1.0 - np.sum(path * phi_full, axis=1)  # per-endogenous disturbance variance
    inv = np.linalg.inv(np.eye(len(endo)) - B)
    corr_eo = Phi_ee @ Gamma.T @ inv.T
    cor_oo = inv @ (Gamma @ Phi_ee @ Gamma.T + np.diag(zeta[endo])) @ inv.T
    np.fill_diagonal(cor_oo, 1.0)

    ordered = exo + endo
    vcv_o = np.block([[Phi_ee, corr_eo], [corr_eo.T, cor_oo]])
    perm = np.argsort(ordered)  # ordered [exo, endo] back to the lv order
    phi = vcv_o[np.ix_(perm, perm)]

    lam = plsc.loadings().reindex(inds).to_numpy(dtype=float)
    lam_mat = np.zeros((len(inds), len(lvs)), dtype=float)
    for a, mv in enumerate(inds):
        lam_mat[a, lvs.index(ind_to_lv[mv])] = lam[a]
    sigma = lam_mat @ phi @ lam_mat.T
    np.fill_diagonal(sigma, 1.0)
    return sigma


def _pair_mask(config: c.Config, inds: list[str], ind_to_lv: dict[str, str]) -> np.ndarray:
    """Strict-lower-triangle mask, excluding within-Mode-B indicator pairs
    (composite indicators are exogenous causes, not common-factor effects —
    same exemption as :class:`.fit.ModelFit`)."""
    n = len(inds)
    tril = np.tri(n, n, k=-1, dtype=bool)
    formative = {lv for lv in {ind_to_lv[i] for i in inds} if config.mode(lv) == Mode.B}
    if not formative:
        return tril
    include = np.ones((n, n), dtype=bool)
    for lv in formative:
        idxs = [i for i, mv in enumerate(inds) if ind_to_lv[mv] == lv]
        for a in idxs:
            for b in idxs:
                include[a, b] = False
    return tril & include


def _discrepancies(S: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    resid = S - sigma
    kept = resid[mask]
    d_uls = float(np.sum(kept ** 2))
    # SRMR normalizes by the number of non-redundant elements p(p+1)/2
    # (including the diagonal, which contributes 0) — matches cSEM.
    p = S.shape[0]
    srmr = float(np.sqrt(d_uls / (p * (p + 1) / 2.0)))
    # geodesic distance: ½ Σ (log₁₀ λ)², λ = eigenvalues of S⁻¹ Σ̂ (cSEM uses base 10).
    try:
        eig = np.real(np.linalg.eigvals(np.linalg.solve(S, sigma)))
        eig = eig[eig > 1e-12]
        d_g = float(0.5 * np.sum(np.log10(eig) ** 2))
    except np.linalg.LinAlgError:
        d_g = float("nan")
    return {"srmr": srmr, "d_uls": d_uls, "d_g": d_g}


class ModelTest:
    """Bollen-Stine bootstrap overall model-fit test (common-factor models)."""

    def __init__(
        self,
        config: c.Config,
        data: pd.DataFrame,
        scheme: Scheme = Scheme.PATH,
        n_boot: int = 499,
        seed: int | None = 1,
    ):
        # Local import to avoid a circular dependency (plspm imports model_test).
        from openpls.plspm import Plspm

        lvs = list(config.path().columns)
        inds: list[str] = []
        ind_to_lv: dict[str, str] = {}
        for lv in lvs:
            for mv in config.mvs(lv):
                if mv in data.columns:
                    inds.append(mv)
                    ind_to_lv[mv] = lv
        if len(inds) < 2:
            raise ValueError("Model test needs at least two indicators.")

        block = data[inds].dropna()
        mask = _pair_mask(config, inds, ind_to_lv)

        def implied(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
            plsc = Plspm(df, config, scheme).plsc()
            S = df[inds].corr().to_numpy(dtype=float)
            sigma = _structural_sigma(plsc, config, lvs, inds, ind_to_lv)
            return S, sigma, plsc.admissibility().admissible

        # Observed discrepancies on the original data.
        S_obs, sigma_obs, _ = implied(block)
        observed = _discrepancies(S_obs, sigma_obs, mask)

        # Bollen-Stine: rotate the standardized indicators so their correlation
        # equals Σ̂ exactly (the "model holds" null population).
        Z = (block[inds].to_numpy(dtype=float) - block[inds].to_numpy(dtype=float).mean(axis=0))
        Z = Z / Z.std(axis=0, ddof=1)
        Z_bs = Z @ _matrix_sqrt(S_obs, inverse=True) @ _matrix_sqrt(sigma_obs)
        z_bs_df = pd.DataFrame(Z_bs, columns=inds)

        rng = np.random.default_rng(seed)
        n = len(block)
        ref: dict[str, list[float]] = {m: [] for m in _MEASURES}
        dropped = 0
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            resample = z_bs_df.iloc[idx].reset_index(drop=True)
            try:
                S_b, sigma_b, admissible = implied(resample)
            except (InadmissiblePLScError, np.linalg.LinAlgError, ValueError):
                dropped += 1
                continue
            # Drop inadmissible resamples (rho_A > 1, non-PD construct matrix,
            # loadings > 1, …) as cSEM does — they produce degenerate Σ̂ that
            # would pollute the reference distribution's tail.
            if not admissible:
                dropped += 1
                continue
            d_b = _discrepancies(S_b, sigma_b, mask)
            if any(not np.isfinite(d_b[m]) for m in _MEASURES):
                dropped += 1
                continue
            for m in _MEASURES:
                ref[m].append(d_b[m])

        results: dict[str, ModelTestMeasure] = {}
        for m in _MEASURES:
            dist = np.array(ref[m], dtype=float)
            if dist.size == 0:
                results[m] = ModelTestMeasure(m, observed[m], float("nan"), float("nan"), float("nan"), False)
                continue
            hi95 = float(np.quantile(dist, 0.95))
            hi99 = float(np.quantile(dist, 0.99))
            p = float((np.sum(dist >= observed[m]) + 1) / (dist.size + 1))
            results[m] = ModelTestMeasure(m, observed[m], hi95, hi99, p, observed[m] > hi95)

        self.__results = results
        self.__n_boot = n_boot
        self.__n_valid = len(ref["srmr"])
        self.__dropped = dropped

    def results(self) -> dict[str, "ModelTestMeasure"]:
        """Per-measure verdicts, keyed by ``srmr`` / ``d_uls`` / ``d_g``."""
        return self.__results

    def rejected(self) -> bool:
        """``True`` if any measure rejects the model at the 5% level."""
        return any(r.rejected for r in self.__results.values())

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "measure": r.measure,
                "observed": r.observed,
                "hi95": r.hi95,
                "hi99": r.hi99,
                "p_value": r.p_value,
                "rejected": r.rejected,
            }
            for r in self.__results.values()
        ]
        return pd.DataFrame(rows).set_index("measure")

    def n_valid_resamples(self) -> int:
        """Number of admissible bootstrap resamples used (inadmissible PLSc
        resamples are dropped, as in cSEM)."""
        return self.__n_valid

    def n_dropped(self) -> int:
        return self.__dropped
