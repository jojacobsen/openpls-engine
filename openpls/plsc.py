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

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.inner_model import _effects
from openpls.mode import Mode
from openpls.specific_indirect import specific_indirect_point


def _rho_a_block(w: np.ndarray, indicators: np.ndarray) -> float:
    """Dijkstra-Henseler rho_A for one Mode-A block.

    Implements the closed-form correction (Dijkstra & Henseler 2015):

        rho_A = (w'w)^2 * (w' S w) / (w' W~ w)

    where ``S`` is the indicator covariance matrix with zero diagonal and
    ``W~ = w w'`` also with zero diagonal. ``indicators`` is the data
    matrix (n × p) for the block; it is z-standardized with Bessel
    correction inside the function. ``w`` is the column vector of outer
    weights (length p).

    Returns 1.0 if the block has fewer than two indicators (rho_A is
    undefined / trivial) or if the formula degenerates (zero denominator).
    """
    p = w.shape[0]
    if p < 2 or indicators.shape[1] != p:
        return 1.0
    if indicators.shape[0] < 2:
        return 1.0
    z = (indicators - indicators.mean(axis=0)) / indicators.std(axis=0, ddof=1)
    s = np.cov(z, rowvar=False, ddof=1)
    np.fill_diagonal(s, 0.0)
    ww = np.outer(w, w)
    np.fill_diagonal(ww, 0.0)
    denom = float(w @ ww @ w)
    if denom == 0.0:
        return 1.0
    wTw = float(w @ w)
    numer = (wTw ** 2) * float(w @ s @ w)
    return numer / denom


class PLSc:
    """Consistent PLS (Dijkstra & Henseler 2015).

    Corrects PLS path coefficients, loadings, and the reflective-LV
    quality criteria for measurement-error attenuation in reflective
    (Mode A) constructs.

    The procedure:

    1. Compute the Dijkstra-Henseler reliability ``rho_A`` for each
       construct. Mode-B (formative) blocks and single-indicator
       constructs get ``rho_A = 1`` (no correction).
    2. Build the construct-correlation matrix from the LV scores and
       dis-attenuate it: divide off-diagonal entries by
       ``sqrt(rho_A_i * rho_A_j)``; diagonal stays 1.
    3. For each endogenous LV, re-estimate the standardized path
       coefficients by OLS on the dis-attenuated correlations, i.e.
       ``beta = R_xx^{-1} r_xy``.
    4. Recompute R² and adjusted R² from the corrected path matrix.
    5. Rescale Mode-A loadings to be consistent with a common-factor
       interpretation: ``lambda_k = w_k * sqrt(rho_A) / (w'w)``.
    6. Recompute the reflective-LV quality panel from the corrected
       loadings and the dis-attenuated correlation matrix: AVE, ρ_c,
       HTMT, SRMR, d_ULS, BIC, and inner VIF. The PLS-SEM
       :class:`~openpls.Plspm` methods keep returning the uncorrected
       composite-model values; the corrected versions live here so a
       caller can keep the two interpretations side-by-side.

    The Mode-A construct scores themselves are *not* changed — only the
    coefficients and quality criteria computed from them. Use the
    corrected outputs when you intend the model to be interpreted as a
    common-factor (covariance-based) model rather than a composite model.
    """

    def __init__(
        self,
        config: c.Config,
        data: pd.DataFrame,
        scores: pd.DataFrame,
        outer_model: pd.DataFrame,
    ):
        path = config.path()
        lvs = list(path.columns)

        rho = pd.Series(1.0, index=lvs, name="rho_a")
        for lv in lvs:
            inds = [mv for mv in config.mvs(lv) if mv in data.columns]
            if config.mode(lv) != Mode.A or len(inds) < 2:
                continue
            block = data[inds].dropna()
            if block.shape[0] < 2:
                continue
            w = outer_model.loc[inds, "weight"].to_numpy(dtype=float)
            rho.loc[lv] = _rho_a_block(w, block.to_numpy(dtype=float))

        adjustment = np.sqrt(np.outer(rho.values, rho.values))
        np.fill_diagonal(adjustment, 1.0)
        raw_cors = scores[lvs].corr().to_numpy()
        adj_cors = pd.DataFrame(raw_cors / adjustment, index=lvs, columns=lvs)

        corrected_paths = pd.DataFrame(0.0, index=path.index, columns=path.columns)
        r_squared = pd.Series(0.0, index=path.index, name="r_squared")
        r_squared_adj = pd.Series(0.0, index=path.index, name="r_squared_adj")
        endogenous: list[str] = []
        n = scores.shape[0]
        for endo in path.index:
            predictors = [lv for lv in path.columns if path.loc[endo, lv] == 1]
            if not predictors:
                continue
            endogenous.append(endo)
            r_xx = adj_cors.loc[predictors, predictors].to_numpy()
            r_xy = adj_cors.loc[predictors, endo].to_numpy()
            beta = np.linalg.solve(r_xx, r_xy)
            corrected_paths.loc[endo, predictors] = beta
            r_sq = float(beta @ r_xy)
            r_squared.loc[endo] = r_sq
            k = len(predictors)
            if n - k - 1 > 0:
                r_squared_adj.loc[endo] = 1 - (1 - r_sq) * (n - 1) / (n - k - 1)
            else:
                r_squared_adj.loc[endo] = np.nan

        loadings = outer_model["loading"].copy()
        for lv in lvs:
            if config.mode(lv) != Mode.A:
                continue
            inds = [mv for mv in config.mvs(lv) if mv in data.columns]
            if len(inds) < 2:
                continue
            w = outer_model.loc[inds, "weight"].to_numpy(dtype=float)
            wTw = float(w @ w)
            if wTw == 0.0:
                continue
            loadings.loc[inds] = w * np.sqrt(rho.loc[lv]) / wTw

        ave = pd.Series(np.nan, index=lvs, name="ave")
        rho_c = pd.Series(np.nan, index=lvs, name="rho_c")
        for lv in lvs:
            if config.mode(lv) != Mode.A:
                continue
            inds = [mv for mv in config.mvs(lv) if mv in data.columns]
            if not inds:
                continue
            lam = loadings.loc[inds].to_numpy(dtype=float)
            if lam.size == 0:
                continue
            sum_sq = float((lam ** 2).sum())
            sum_uniq = float((1.0 - lam ** 2).sum())
            denom_ave = sum_sq + sum_uniq
            if denom_ave > 0:
                ave.loc[lv] = sum_sq / denom_ave
            sum_lam = float(lam.sum())
            denom_rho = (sum_lam ** 2) + sum_uniq
            if denom_rho > 0:
                rho_c.loc[lv] = (sum_lam ** 2) / denom_rho

        htmt_values = np.abs(adj_cors.to_numpy(copy=True))
        np.fill_diagonal(htmt_values, np.nan)
        htmt = pd.DataFrame(htmt_values, index=adj_cors.index, columns=adj_cors.columns)

        srmr, d_uls = self.__compute_fit(config, data, lvs, loadings, adj_cors)

        bic = pd.Series(np.nan, index=lvs, name="bic")
        for endo in endogenous:
            k = int(path.loc[endo].sum())
            r2 = float(r_squared.loc[endo])
            if k > 0 and n > k + 1:
                sse = max((1.0 - r2) * (n - 1), 1e-12)
                bic.loc[endo] = n * np.log(sse / n) + (k + 1) * np.log(n)

        vif_inner = self.__compute_inner_vif(path, adj_cors)

        self.__path = path
        self.__rho = rho
        self.__adj_cors = adj_cors
        self.__paths = corrected_paths
        self.__r_squared = r_squared
        self.__r_squared_adj = r_squared_adj
        self.__loadings = loadings.rename("loading_c")
        self.__endogenous = endogenous
        self.__ave = ave
        self.__rho_c = rho_c
        self.__htmt = htmt
        self.__srmr = srmr
        self.__d_uls = d_uls
        self.__bic = bic
        self.__vif_inner = vif_inner
        self.__effects = _effects(corrected_paths)

    @staticmethod
    def __compute_fit(
        config: c.Config,
        data: pd.DataFrame,
        lvs: list[str],
        loadings: pd.Series,
        adj_cors: pd.DataFrame,
    ) -> tuple[float, float]:
        ind_to_lv: dict[str, str] = {}
        inds: list[str] = []
        for lv in lvs:
            for mv in config.mvs(lv):
                if mv in data.columns and mv in loadings.index:
                    ind_to_lv[mv] = lv
                    inds.append(mv)
        if len(inds) < 2:
            return float("nan"), float("nan")
        n_ind, n_lv = len(inds), len(lvs)
        lam = loadings.loc[inds].to_numpy(dtype=float)
        lv_idx = np.array([lvs.index(ind_to_lv[i]) for i in inds])
        Lambda = np.zeros((n_ind, n_lv), dtype=float)
        Lambda[np.arange(n_ind), lv_idx] = lam
        phi = adj_cors.to_numpy()
        s = data[inds].corr().to_numpy()
        if np.isnan(phi).any() or np.isnan(s).any():
            return float("nan"), float("nan")
        implied = Lambda @ phi @ Lambda.T
        np.fill_diagonal(implied, 1.0)
        resid = s - implied
        formative_lvs = {lv for lv in lvs if config.mode(lv) == Mode.B}
        if formative_lvs:
            include = np.ones((n_ind, n_ind), dtype=bool)
            for lv in formative_lvs:
                idxs = [i for i, ind in enumerate(inds) if ind_to_lv[ind] == lv]
                for a in idxs:
                    for b in idxs:
                        include[a, b] = False
            tril_mask = np.tri(n_ind, n_ind, k=-1, dtype=bool)
            kept = resid[tril_mask & include]
        else:
            kept = resid[np.tril_indices(n_ind, k=-1)]
        if kept.size == 0:
            return float("nan"), float("nan")
        srmr = float(np.sqrt(np.mean(kept ** 2)))
        d_uls = float(np.sum(kept ** 2))
        return srmr, d_uls

    @staticmethod
    def __compute_inner_vif(
        path: pd.DataFrame, adj_cors: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for endo in path.index:
            predictors = [lv for lv in path.columns if path.loc[endo, lv] == 1]
            if len(predictors) < 2:
                continue
            rows: list[dict] = []
            for j, j_lv in enumerate(predictors):
                others = [p for i, p in enumerate(predictors) if i != j]
                r_oo = adj_cors.loc[others, others].to_numpy()
                r_jo = adj_cors.loc[others, j_lv].to_numpy()
                try:
                    beta = np.linalg.solve(r_oo, r_jo)
                except np.linalg.LinAlgError:
                    rows.append({"predictor": j_lv, "vif": float("nan")})
                    continue
                r2 = float(r_jo @ beta)
                if r2 >= 1.0 - 1e-12:
                    rows.append({"predictor": j_lv, "vif": float("inf")})
                    continue
                if r2 < 0.0:
                    r2 = 0.0
                rows.append({"predictor": j_lv, "vif": 1.0 / (1.0 - r2)})
            if rows:
                out[endo] = pd.DataFrame(rows, columns=["predictor", "vif"])
        return out

    def rho_a(self) -> pd.Series:
        """Dijkstra-Henseler ``rho_A`` per construct.

        Reflective constructs with at least two indicators receive the
        bias-corrected reliability estimate. Formative and single-item
        constructs receive ``1.0`` by convention (no correction is
        applied to them).
        """
        return self.__rho

    def adjusted_correlations(self) -> pd.DataFrame:
        """Dis-attenuated construct correlation matrix used by PLSc.

        Each off-diagonal cell ``r(i, j)`` is the raw construct-score
        correlation divided by ``sqrt(rho_A_i * rho_A_j)``. The diagonal
        is forced to 1.
        """
        return self.__adj_cors

    def path_coefficients(self) -> pd.DataFrame:
        """Corrected standardized path coefficients.

        Same shape as :meth:`Plspm.path_coefficients`, but recomputed by
        OLS on the dis-attenuated construct correlation matrix.
        """
        return self.__paths

    def r_squared(self) -> pd.Series:
        """Corrected R² per endogenous LV (from the adjusted paths)."""
        return self.__r_squared

    def r_squared_adj(self) -> pd.Series:
        """Corrected adjusted R² per endogenous LV."""
        return self.__r_squared_adj

    def effects(self) -> pd.DataFrame:
        """Direct, indirect, and total effects from the corrected path matrix.

        Mirrors :meth:`Plspm.effects` but multiplies the dis-attenuated
        PLSc path coefficients along the structural DAG instead of the
        uncorrected composite ones. For a chain ``A -> M -> Y``, the
        indirect effect contributed by that chain is the product of the
        PLSc β along the chain; the total effect from A to Y is the
        sum of all such products (including the direct edge A -> Y if
        present). Use these when the model is being interpreted as a
        common-factor model.

        Returns:
            DataFrame indexed by ``"<from> -> <to>"`` with columns
            ``from``, ``to``, ``direct``, ``indirect``, ``total``.
        """
        return self.__effects

    def specific_indirect_effects(
        self,
        source: str,
        target: str,
        through: list[str] | None = None,
    ) -> pd.DataFrame:
        """Point-estimate specific indirect effects on the corrected paths.

        Same interface and semantics as
        :meth:`Plspm.specific_indirect_effects`, but multiplies the
        dis-attenuated PLSc β along each mediation chain instead of the
        uncorrected composite β. Required when the structural model is
        interpreted as a common-factor (covariance-based) model — the
        composite-model SIE would otherwise carry the PLS-SEM
        measurement-error attenuation forward into the mediation
        estimate.

        Args:
            source: source LV name.
            target: target LV name.
            through: explicit mediator chain (e.g. ``["M1", "M2"]``) or
                ``None`` to enumerate every chain ``source -> ... -> target``.

        Returns:
            DataFrame indexed by chain label (``"A -> M -> Y"``) with
            columns ``from``, ``to``, ``via`` (tuple of intermediates),
            and ``estimate`` (product of corrected β along the chain).

        Raises:
            ValueError: if ``source == target``, no indirect chain
                exists, or ``through`` references nonexistent edges.
            KeyError: if ``source`` or ``target`` is not in the model.
        """
        return specific_indirect_point(
            self.__paths,
            self.__path,
            source,
            target,
            through=through,
        )

    def loadings(self) -> pd.Series:
        """Corrected outer loadings (common-factor interpretation).

        For each Mode-A block, indicator ``k`` is rescaled to
        ``w_k * sqrt(rho_A) / (w'w)``. Mode-B and single-indicator
        constructs keep their PLS loadings unchanged.
        """
        return self.__loadings

    def ave(self) -> pd.Series:
        """Corrected Average Variance Extracted per Mode-A LV.

        Computed from the PLSc loadings: ``AVE_c = mean(λ_c²)`` where
        ``λ_c = w · sqrt(rho_A) / (w'w)``. ``NaN`` for Mode-B
        (formative) and single-indicator constructs (AVE is undefined).
        """
        return self.__ave

    def rho_c(self) -> pd.Series:
        """Corrected composite reliability (Jöreskog's ρ) per Mode-A LV.

        ``ρ_c = (Σ λ_c)² / ((Σ λ_c)² + Σ(1 - λ_c²))`` evaluated on the
        PLSc loadings. ``NaN`` for Mode-B and single-indicator
        constructs (where the formula is undefined / trivially 1).
        """
        return self.__rho_c

    def htmt(self) -> pd.DataFrame:
        """PLSc-consistent HTMT — the disattenuated construct
        correlation magnitude.

        Under congeneric reflective measurement, the disattenuated
        correlation ``R̃(i, j) = corr(η̂_i, η̂_j) / sqrt(rho_A_i · rho_A_j)``
        is a consistent estimator of the latent correlation, which is the
        quantity HTMT itself targets (Henseler, Ringle & Sarstedt 2015).
        Returns ``|R̃(i, j)|`` as a square matrix with ``NaN`` on the
        diagonal. The same conservative discriminant-validity thresholds
        as HTMT apply (``< 0.85`` / ``< 0.90``).
        """
        return self.__htmt

    def srmr(self) -> float:
        """SRMR computed from the PLSc model-implied correlations.

        Builds ``Σ̂_c = Λ_c · Φ_c · Λ_cᵀ`` with diagonal forced to 1,
        where ``Λ_c`` is the corrected indicator-loading matrix and
        ``Φ_c`` the dis-attenuated construct correlation matrix.
        ``SRMR_c = sqrt(mean residual² over the strict lower triangle)``,
        excluding within-Mode-B pairs the same way :class:`.fit.ModelFit`
        does.
        """
        return self.__srmr

    def d_uls(self) -> float:
        """d_ULS on the PLSc residuals.

        Sum of squared residuals between the observed indicator
        correlations and the PLSc model-implied ``Σ̂_c`` over the strict
        lower triangle (same pair-mask as :meth:`srmr`).
        """
        return self.__d_uls

    def bic(self) -> pd.Series:
        """Bayesian Information Criterion on the corrected R² per endogenous LV.

        ``BIC_c = n · log(SSE_c / n) + (k + 1) · log(n)`` with
        ``SSE_c = (1 - R²_c) · (n - 1)``. The corrected R² is generally
        larger than the uncorrected one, so ``BIC_c`` is generally
        smaller. ``NaN`` for exogenous LVs.
        """
        return self.__bic

    def vif_inner(self) -> dict[str, pd.DataFrame]:
        """Inner VIF per endogenous LV computed on the dis-attenuated
        construct correlations.

        For each endogenous LV with two or more predictors, regresses
        each predictor on the others in the dis-attenuated correlation
        metric and reports ``VIF = 1 / (1 - R²)``. Endogenous LVs with
        fewer than two predictors are omitted (VIF is trivially 1).
        """
        return self.__vif_inner

    def summary(self) -> pd.DataFrame:
        """Per-LV summary: ``rho_A``, AVE, ρ_c, corrected R² / adj R², BIC."""
        out = pd.DataFrame({"rho_a": self.__rho})
        out["ave"] = self.__ave.reindex(out.index)
        out["rho_c"] = self.__rho_c.reindex(out.index)
        out["r_squared"] = self.__r_squared.reindex(out.index)
        out["r_squared_adj"] = self.__r_squared_adj.reindex(out.index)
        out["bic"] = self.__bic.reindex(out.index)
        out.loc[~out.index.isin(self.__endogenous), ["r_squared", "r_squared_adj", "bic"]] = np.nan
        return out
