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


def _vif_one(y: np.ndarray, x: np.ndarray) -> float:
    """OLS auxiliary regression of ``y`` on ``x`` (with intercept) → VIF.

    VIF = 1 / (1 - R²). Returns ``inf`` when R² is numerically 1 (perfect
    collinearity). Returns ``nan`` when ``y`` has zero variance or the
    regression is otherwise degenerate.
    """
    n = y.shape[0]
    if n < 2 or x.shape[0] != n or x.shape[1] == 0:
        return float("nan")
    y_mean = float(y.mean())
    ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    design = np.column_stack([np.ones(n), x])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y - design @ beta
    ss_res = float((resid ** 2).sum())
    r_sq = 1.0 - ss_res / ss_tot
    # Numerical noise can push R² fractionally above 1 or below 0.
    if r_sq >= 1.0 - 1e-12:
        return float("inf")
    if r_sq < 0.0:
        r_sq = 0.0
    return 1.0 / (1.0 - r_sq)


class VIF:
    """Variance Inflation Factor diagnostics for the outer and inner model.

    Two views are available:

    * :meth:`items` — per-indicator VIF within each construct block. For
      every indicator ``x_j`` in a block with ≥ 2 indicators, ``x_j`` is
      regressed on the remaining indicators of the same block and
      ``VIF_j = 1 / (1 - R²_j)``. Standard collinearity diagnostic for
      formative (Mode B) blocks; also informative for reflective blocks.
    * :meth:`inner` — per-predictor VIF for each endogenous latent
      variable. For every predictor LV of an endogenous LV ``Y``, the
      predictor's score is regressed on the other predictors' scores and
      ``VIF`` is reported. Use to detect structural multicollinearity
      among antecedents.

    A common rule of thumb is ``VIF < 5`` (lenient) or ``< 3.3``
    (Diamantopoulos & Siguaw 2006) for formative indicators.
    """

    def __init__(self, config: c.Config, data: pd.DataFrame, scores: pd.DataFrame):
        self.__items = self.__compute_items(config, data)
        self.__inner = self.__compute_inner(config, scores)

    @staticmethod
    def __compute_items(config: c.Config, data: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for lv in config.path().columns:
            inds = [mv for mv in config.mvs(lv) if mv in data.columns]
            if len(inds) < 2:
                continue
            block = data[inds].dropna()
            if block.shape[0] < 2:
                continue
            block_np = block.to_numpy(dtype=float)
            for j, ind in enumerate(inds):
                y = block_np[:, j]
                x = np.delete(block_np, j, axis=1)
                rows.append({"lv": lv, "indicator": ind, "vif": _vif_one(y, x)})
        return pd.DataFrame(rows, columns=["lv", "indicator", "vif"])

    @staticmethod
    def __compute_inner(
        config: c.Config, scores: pd.DataFrame
    ) -> dict[str, pd.DataFrame]:
        path = config.path()
        out: dict[str, pd.DataFrame] = {}
        for endo in path.index:
            predictors = [lv for lv in path.columns if path.loc[endo, lv] == 1]
            if len(predictors) < 2:
                continue
            block = scores[predictors].dropna()
            if block.shape[0] < 2:
                continue
            block_np = block.to_numpy(dtype=float)
            rows: list[dict] = []
            for j, lv in enumerate(predictors):
                y = block_np[:, j]
                x = np.delete(block_np, j, axis=1)
                rows.append({"predictor": lv, "vif": _vif_one(y, x)})
            out[endo] = pd.DataFrame(rows, columns=["predictor", "vif"])
        return out

    def items(self) -> pd.DataFrame:
        """Per-indicator VIF within each construct block.

        Long format with columns ``lv``, ``indicator``, ``vif``. Blocks
        with fewer than two indicators are omitted (VIF is undefined).
        Two-indicator blocks resolve to ``VIF = 1 / (1 - r²)`` where ``r``
        is the within-block correlation.
        """
        return self.__items

    def inner(self) -> dict[str, pd.DataFrame]:
        """Per-predictor VIF for each endogenous latent variable.

        Returns a dict keyed by endogenous LV name. Each value is a
        DataFrame with columns ``predictor`` and ``vif``. Endogenous LVs
        with fewer than two predictors are omitted (VIF is trivially 1).
        """
        return self.__inner
