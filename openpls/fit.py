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


class ModelFit:
    """Model-fit indices comparing observed vs model-implied indicator correlations.

    The saturated model-implied correlation is

        Σ̂ = Λ Φ Λᵀ  with diag(Σ̂) = 1

    where Λ is the n × k indicator-loading matrix (one non-zero per row,
    keyed to the LV the indicator belongs to) and Φ is the k × k LV
    correlation matrix. SRMR averages squared residuals over the strict
    lower triangle (the diagonal is forced to 1 by construction);
    d_ULS is the unweighted sum of those squared residuals.

    See Henseler et al. (2014) for the construction; Henseler et al. (2014)
    suggest SRMR < 0.08 as an acceptable-fit threshold.
    """

    def __init__(self, config: c.Config, data: pd.DataFrame, scores: pd.DataFrame, outer_model: pd.DataFrame):
        # Build the indicator → LV map and the ordered indicator list
        # following the path-matrix LV order (which is the same order as
        # `scores.columns` produced by the estimator).
        lv_names = list(scores.columns)
        ind_to_lv: dict[str, str] = {}
        inds: list[str] = []
        for lv in lv_names:
            for mv in config.mvs(lv):
                if mv in data.columns and mv in outer_model.index:
                    ind_to_lv[mv] = lv
                    inds.append(mv)

        if len(inds) < 2:
            self.__srmr = float("nan")
            self.__d_uls = float("nan")
            self.__residuals = pd.DataFrame()
            return

        n_ind, n_lv = len(inds), len(lv_names)
        loadings = outer_model.loc[inds, "loading"].to_numpy(dtype=float)
        lv_idx = np.array([lv_names.index(ind_to_lv[i]) for i in inds])
        Lambda = np.zeros((n_ind, n_lv), dtype=float)
        Lambda[np.arange(n_ind), lv_idx] = loadings

        Phi = scores.corr().reindex(index=lv_names, columns=lv_names).to_numpy()
        S = data[inds].corr().to_numpy()

        if np.isnan(Phi).any() or np.isnan(S).any():
            self.__srmr = float("nan")
            self.__d_uls = float("nan")
            self.__residuals = pd.DataFrame()
            return

        implied = Lambda @ Phi @ Lambda.T
        np.fill_diagonal(implied, 1.0)
        resid = S - implied

        iu = np.tril_indices(n_ind, k=-1)
        self.__srmr = float(np.sqrt(np.mean(resid[iu] ** 2)))
        self.__d_uls = float(np.sum(resid[iu] ** 2))
        self.__residuals = pd.DataFrame(resid, index=inds, columns=inds)

    def srmr(self) -> float:
        """Standardized Root Mean Square Residual.

        ``< 0.08`` is the conventional acceptable-fit threshold
        (Henseler et al., 2014).
        """
        return self.__srmr

    def d_uls(self) -> float:
        """d_ULS — squared Euclidean distance between observed and
        model-implied indicator correlation matrices (strict lower triangle).
        """
        return self.__d_uls

    def residuals(self) -> pd.DataFrame:
        """Indicator-correlation residual matrix (observed − model-implied).

        Returns:
            a square DataFrame indexed by indicator name. Empty if the fit
            could not be computed (e.g. fewer than two indicators).
        """
        return self.__residuals

    def summary(self) -> pd.DataFrame:
        """One-row summary of the fit indices."""
        return pd.DataFrame(
            {"srmr": [self.__srmr], "d_uls": [self.__d_uls]},
            index=["saturated"],
        )
