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
from openpls.mode import Mode


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

    Corrects PLS path coefficients, loadings, and R² for measurement-
    error attenuation in reflective (Mode A) constructs.

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

    The Mode-A construct scores themselves are *not* changed — only the
    coefficients computed from them. Use the corrected outputs when you
    intend the model to be interpreted as a common-factor (covariance-
    based) model rather than a composite model.
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

        self.__rho = rho
        self.__adj_cors = adj_cors
        self.__paths = corrected_paths
        self.__r_squared = r_squared
        self.__r_squared_adj = r_squared_adj
        self.__loadings = loadings.rename("loading_c")
        self.__endogenous = endogenous

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

    def loadings(self) -> pd.Series:
        """Corrected outer loadings (common-factor interpretation).

        For each Mode-A block, indicator ``k`` is rescaled to
        ``w_k * sqrt(rho_A) / (w'w)``. Mode-B and single-indicator
        constructs keep their PLS loadings unchanged.
        """
        return self.__loadings

    def summary(self) -> pd.DataFrame:
        """Per-LV summary: ``rho_A`` and (when endogenous) corrected R² / adj R²."""
        out = pd.DataFrame({"rho_a": self.__rho})
        out["r_squared"] = self.__r_squared.reindex(out.index)
        out["r_squared_adj"] = self.__r_squared_adj.reindex(out.index)
        out.loc[~out.index.isin(self.__endogenous), ["r_squared", "r_squared_adj"]] = np.nan
        return out
