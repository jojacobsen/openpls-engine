#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
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
from sklearn.decomposition import PCA

import plspm.util as util
from plspm.config import Config
from plspm.mode import Mode


class Unidimensionality:
    """Internal class that computes various reliability metrics. Use the method :meth:`~.plspm.Plspm.unidimensionality` defined on :class:`~.plspm.Plspm` to retrieve the results."""
    def __init__(self, config: Config, data: pd.DataFrame, correction: float):
        self.__config = config
        self.__data = data
        self.__correction = correction

    def summary(self):
        """Internal method that performs principal component analysis to compute various reliability metrics.

        For latent variables whose indicator block contains missing values, the
        computation falls back to listwise deletion (rows with any NaN in the
        block are dropped) with a per-block correction factor. Upstream `plspm`
        skipped the LV entirely in this case (OpenPLS fix #65).
        """
        summary = pd.DataFrame({"mode":                 pd.Series(dtype="str"),
                                "mvs":                  pd.Series(dtype="float"),
                                "cronbach_alpha":       pd.Series(dtype="float"),
                                "dillon_goldstein_rho": pd.Series(dtype="float"),
                                "eig_1st":              pd.Series(dtype="float"),
                                "eig_2nd":              pd.Series(dtype="float")},
                                 index=list(self.__config.path()))
        for lv in list(self.__config.path()):
            mvs = len(self.__config.mvs(lv))
            summary.loc[lv, "mode"] = self.__config.mode(lv).name
            summary.loc[lv, "mvs"] = mvs
            block = self.__data.loc[:, self.__config.mvs(lv)]
            if block.isnull().values.any():
                block = block.dropna()
                if block.shape[0] < 2:
                    continue
                correction = np.sqrt(block.shape[0] / (block.shape[0] - 1))
            else:
                correction = self.__correction
            mvs_for_lvs = util.treat(block) * correction
            pca_input = mvs_for_lvs if mvs_for_lvs.shape[0] > mvs_for_lvs.shape[1] else mvs_for_lvs.transpose()
            pca = PCA()
            pca_scores = pca.fit_transform(pca_input)
            pca_std_dev = np.std(pca_scores, axis=0)
            summary.loc[lv, "eig_1st"] = pca_std_dev[0] ** 2
            summary.loc[lv, "eig_2nd"] = pca_std_dev[1] ** 2 if mvs > 1 else np.nan
            if (self.__config.mode(lv) == Mode.A):
                if mvs > 1:
                    ca_numerator = 2 * np.tril(pca_input.corr(), -1).sum()
                    ca_denominator = pca_input.sum(axis=1).var() / correction ** 2
                    ca = max(0, (ca_numerator / ca_denominator) * (mvs / (mvs - 1)))
                else:
                    ca = np.nan
                summary.loc[lv, "cronbach_alpha"] = ca
                corr = np.corrcoef(np.column_stack((pca_input.values, pca_scores[:,0])), rowvar=False)[:,-1][:-1]
                rho_numerator = sum(corr) ** 2
                rho_denominator = rho_numerator + (mvs - np.sum(np.power(corr, 2)))
                summary.loc[lv, "dillon_goldstein_rho"] = rho_numerator / rho_denominator
        return summary
