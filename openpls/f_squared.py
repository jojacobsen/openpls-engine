#!/usr/bin/python3
#
# Copyright (C) 2026 Johannes Jacob
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

"""Cohen's f² effect size for structural-model predictors.

For every directed edge ``predictor -> endogenous`` in the structural
model, ``f² = (R²_full − R²_reduced) / (1 − R²_full)`` measures the
proportionate change in R² when the predictor is removed (Cohen 1988;
Hair, Hult, Ringle & Sarstedt 2022). Conventional thresholds:

- ``f² >= 0.02`` — small effect
- ``f² >= 0.15`` — medium effect
- ``f² >= 0.35`` — large effect
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from openpls.config import Config


def _effect_size(value: float) -> str:
    if not np.isfinite(value) or value < 0.02:
        return "none"
    if value < 0.15:
        return "small"
    if value < 0.35:
        return "medium"
    return "large"


class FSquared:
    """Cohen's f² effect size per structural-model edge.

    Constructed lazily on first call to :meth:`~openpls.Plspm.f_squared`.
    For each endogenous LV and each predictor in its structural equation,
    the LV's OLS is refit on the standardized scores with that predictor
    removed; the resulting reduced R² is combined with the full R² to
    give f². Endogenous LVs with a single predictor return f² = R²_full /
    (1 − R²_full) (reduced model is intercept-only on a standardized
    response, R²_red = 0).
    """

    def __init__(
        self,
        config: Config,
        scores: pd.DataFrame,
        path_coefficients: pd.DataFrame,
        r_squared: pd.Series,
    ):
        path = config.path()
        rows = []
        endogenous = path.sum(axis=1).astype(bool)
        endogenous_lvs = list(endogenous[endogenous].index)
        for lv in endogenous_lvs:
            predictors = list(path.loc[lv][path.loc[lv] == 1].index)
            r2_full = float(r_squared.loc[lv])
            for predictor in predictors:
                reduced = [p for p in predictors if p != predictor]
                if reduced:
                    exog = sm.add_constant(scores.loc[:, reduced])
                    r2_red = float(sm.OLS(scores.loc[:, lv], exog).fit().rsquared)
                else:
                    r2_red = 0.0
                if r2_full >= 1.0 - 1e-12:
                    f2 = np.inf
                else:
                    f2 = (r2_full - r2_red) / (1.0 - r2_full)
                rows.append({
                    "from": predictor,
                    "to": lv,
                    "r_squared_full": r2_full,
                    "r_squared_reduced": r2_red,
                    "f_squared": f2,
                    "effect_size": _effect_size(f2),
                })
        self.__table = pd.DataFrame(
            rows,
            index=[f"{r['from']} -> {r['to']}" for r in rows],
        )
        self.__matrix = self.__build_matrix(path, self.__table)

    @staticmethod
    def __build_matrix(path: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
        m = pd.DataFrame(np.nan, index=path.index, columns=path.columns)
        for _, row in table.iterrows():
            m.loc[row["to"], row["from"]] = row["f_squared"]
        return m

    def table(self) -> pd.DataFrame:
        """Long-format table indexed by ``"predictor -> endogenous"``.

        Columns: ``from``, ``to``, ``r_squared_full``,
        ``r_squared_reduced``, ``f_squared``, ``effect_size``.
        """
        return self.__table

    def matrix(self) -> pd.DataFrame:
        """Square matrix of f² values mirroring the path matrix.

        Rows are targets (endogenous LVs), columns are sources. Cells
        without a structural path are ``NaN``.
        """
        return self.__matrix
