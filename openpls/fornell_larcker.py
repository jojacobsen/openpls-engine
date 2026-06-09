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

"""Fornell-Larcker discriminant-validity criterion.

The classical Fornell and Larcker (1981) check: for each reflective
(Mode A) latent variable, the square root of its AVE must exceed its
absolute correlation with every other latent variable. The matrix
returned here has ``sqrt(AVE)`` on the diagonal and the inter-construct
correlations off-diagonal. Formative (Mode B) and single-indicator LVs
receive ``NaN`` on the diagonal because AVE is undefined for them; the
modern recommendation (Henseler, Ringle and Sarstedt 2015) is to use
HTMT for discriminant validity, so this criterion is provided alongside
HTMT, not in place of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpls.config import Config
from openpls.mode import Mode


def _ave(config: Config, outer_model: pd.DataFrame) -> pd.Series:
    """AVE per Mode-A LV (matches inner_summary.InnerSummary)."""
    ave = pd.Series(np.nan, index=config.path().index, name="ave")
    for lv in list(config.path()):
        if config.mode(lv) != Mode.A:
            continue
        communality = outer_model.loc[:, "communality"].loc[config.mvs(lv)]
        num = communality.sum()
        denom = num + (1 - communality).sum()
        if denom > 0:
            ave.loc[lv] = num / denom
    return ave


class FornellLarcker:
    """Fornell-Larcker discriminant-validity matrix and verdict.

    Constructed lazily on first call to
    :meth:`~openpls.Plspm.fornell_larcker`. The criterion is satisfied
    for a reflective LV when the diagonal entry ``sqrt(AVE_lv)`` exceeds
    every absolute off-diagonal entry in the same row.
    """

    def __init__(
        self,
        config: Config,
        scores: pd.DataFrame,
        outer_model: pd.DataFrame,
    ):
        ave = _ave(config, outer_model)
        lvs = list(config.path().index)
        # Inter-LV correlations from the standardized scores.
        corr = scores.loc[:, lvs].corr()
        matrix = corr.copy()
        for lv in lvs:
            matrix.loc[lv, lv] = float(np.sqrt(ave.loc[lv])) if not np.isnan(ave.loc[lv]) else np.nan
        self.__matrix = matrix
        self.__ave = ave
        self.__summary = self.__build_summary(matrix, ave, lvs)

    @staticmethod
    def __build_summary(matrix: pd.DataFrame, ave: pd.Series, lvs: list[str]) -> pd.DataFrame:
        rows = []
        for lv in lvs:
            sqrt_ave = matrix.loc[lv, lv]
            if np.isnan(sqrt_ave):
                rows.append({
                    "lv": lv,
                    "sqrt_ave": np.nan,
                    "max_abs_corr": np.nan,
                    "passes": pd.NA,
                    "note": "no AVE (non Mode-A or single-indicator)",
                })
                continue
            off = matrix.loc[lv, :].drop(lv).abs()
            max_off = float(off.max()) if len(off) else 0.0
            rows.append({
                "lv": lv,
                "sqrt_ave": float(sqrt_ave),
                "max_abs_corr": max_off,
                "passes": bool(sqrt_ave > max_off),
                "note": "",
            })
        return pd.DataFrame(rows).set_index("lv")

    def matrix(self) -> pd.DataFrame:
        """Square Fornell-Larcker matrix.

        Diagonal: ``sqrt(AVE)`` per LV (``NaN`` for Mode B / single-
        indicator). Off-diagonal: latent-variable correlations from the
        standardized scores.
        """
        return self.__matrix

    def ave(self) -> pd.Series:
        """Average Variance Extracted per LV (``NaN`` for Mode B /
        single-indicator)."""
        return self.__ave

    def summary(self) -> pd.DataFrame:
        """Per-LV discriminant-validity verdict.

        Columns: ``sqrt_ave``, ``max_abs_corr``, ``passes`` (boolean,
        ``NA`` when AVE is undefined), ``note``.
        """
        return self.__summary
