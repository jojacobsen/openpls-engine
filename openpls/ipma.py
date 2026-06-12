#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
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

"""Importance-Performance Map Analysis (IPMA).

For a chosen target endogenous LV, IPMA reports each predecessor LV's
importance (total effect on the target) and performance (mean of its
0-100-rescaled latent-variable score). The map highlights LVs with high
importance but low performance as priority improvement targets.

References
----------
- Ringle, C. M., & Sarstedt, M. (2016). Gain more insight from your
  PLS-SEM results: The importance-performance map analysis.
  Industrial Management & Data Systems, 116(9), 1865-1886.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2017).
  A primer on partial least squares structural equation modeling
  (PLS-SEM), 2nd ed., Chapter 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openpls.config import Config


def _rescale(x: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo or np.isnan(lo) or np.isnan(hi):
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x.astype(float) - lo) / (hi - lo) * 100.0


class IPMA:
    """Importance-Performance Map Analysis for one target endogenous LV.

    Parameters
    ----------
    config : Config
        The model config used in the original Plspm fit.
    data : pd.DataFrame
        The indicator data the engine actually saw (after missing-value
        handling). This is the data whose rescaled values define
        performance.
    outer_weights : pd.DataFrame
        Outer model from ``Plspm.outer_model()``; must contain a ``weight``
        column indexed by indicator name.
    effects : pd.DataFrame
        Path effects from ``Plspm.effects()``; must contain ``from``,
        ``to``, ``total`` columns.
    target : str
        Endogenous LV to analyze.
    scale_min, scale_max : float, optional
        Common scale bounds for rescaling all indicators (e.g. 1 and 7 for
        a 7-point Likert). If both are ``None``, each indicator is rescaled
        from its observed min/max in ``data``.
    indicator_scales : dict[str, tuple[float, float]], optional
        Per-indicator override of ``(min, max)``. Takes precedence over
        the global ``scale_min``/``scale_max``.
    """

    def __init__(
        self,
        config: Config,
        data: pd.DataFrame,
        outer_weights: pd.DataFrame,
        effects: pd.DataFrame,
        target: str,
        scale_min: float | None = None,
        scale_max: float | None = None,
        indicator_scales: dict[str, tuple[float, float]] | None = None,
    ):
        path = config.path()
        if target not in path.index:
            raise ValueError(f"target {target!r} is not a latent variable in the model")
        if path.loc[target].sum() == 0:
            raise ValueError(
                f"target {target!r} is exogenous; IPMA requires an endogenous target"
            )
        if (scale_min is None) != (scale_max is None):
            raise ValueError("scale_min and scale_max must be provided together or both None")
        if scale_min is not None and scale_max is not None and scale_max <= scale_min:
            raise ValueError("scale_max must be greater than scale_min")
        self.__target = target
        self.__config = config
        self.__data = data
        self.__weights = outer_weights
        self.__effects = effects
        self.__scale_min = scale_min
        self.__scale_max = scale_max
        self.__overrides = indicator_scales or {}
        self.__lv_table: pd.DataFrame | None = None
        self.__indicator_table: pd.DataFrame | None = None

    @property
    def target(self) -> str:
        return self.__target

    def __indicator_bounds(self, indicator: str) -> tuple[float, float]:
        if indicator in self.__overrides:
            lo, hi = self.__overrides[indicator]
            return float(lo), float(hi)
        if self.__scale_min is not None and self.__scale_max is not None:
            return float(self.__scale_min), float(self.__scale_max)
        col = pd.to_numeric(self.__data[indicator], errors="coerce")
        return float(col.min(skipna=True)), float(col.max(skipna=True))

    def __compute(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        path = self.__config.path()
        target = self.__target
        eff = self.__effects
        relevant = (
            eff[eff["to"] == target].set_index("from")
            if not eff.empty
            else pd.DataFrame(columns=["total"])
        )
        lv_imp: dict[str, float] = {
            lv: float(relevant.loc[lv, "total"]) for lv in relevant.index
        }

        ind_rows: list[dict] = []
        lv_perf: dict[str, float] = {}
        for lv in path.index:
            inds = [i for i in self.__config.mvs(lv) if i in self.__data.columns]
            if not inds:
                continue
            rescaled = pd.DataFrame(index=self.__data.index)
            bounds: dict[str, tuple[float, float]] = {}
            for ind in inds:
                lo, hi = self.__indicator_bounds(ind)
                bounds[ind] = (lo, hi)
                rescaled[ind] = _rescale(self.__data[ind], lo, hi)
            w_raw = pd.Series(
                {
                    ind: float(self.__weights.loc[ind, "weight"])
                    for ind in inds
                    if ind in self.__weights.index
                },
                dtype=float,
            )
            if w_raw.empty or w_raw.abs().sum() == 0:
                continue
            denom = w_raw.sum()
            if denom == 0:
                # mixed-sign block; fall back to absolute-sum normalization
                w_norm = w_raw / w_raw.abs().sum()
            else:
                w_norm = w_raw / denom
            block = rescaled[w_norm.index]
            perf_series = block.mul(w_norm, axis=1).sum(axis=1, skipna=False)
            valid = block.notna().all(axis=1)
            mean_perf = float(perf_series[valid].mean()) if valid.any() else float("nan")
            lv_perf[lv] = mean_perf
            lv_total_effect = lv_imp.get(lv, float("nan"))
            for ind in w_raw.index:
                lo, hi = bounds[ind]
                series = rescaled[ind].dropna()
                mean_rescaled = float(series.mean()) if not series.empty else float("nan")
                outer_w = float(w_raw.loc[ind])
                ind_importance = (
                    outer_w * lv_total_effect
                    if not np.isnan(lv_total_effect)
                    else float("nan")
                )
                # Henseler IPMA normalized weight (Ringle & Sarstedt 2016):
                # the indicator's share of its LV's importance on the target,
                # i.e. (outer_weight × lv_importance) / lv_importance = outer
                # weight. Computed as the explicit ratio so consumers can
                # reproduce the SmartPLS-style indicator-importance table.
                if np.isnan(lv_total_effect) or lv_total_effect == 0:
                    henseler_norm = float("nan")
                else:
                    henseler_norm = ind_importance / lv_total_effect
                ind_rows.append(
                    {
                        "lv": lv,
                        "indicator": ind,
                        "outer_weight": outer_w,
                        "normalized_weight": float(w_norm.loc[ind]),
                        "indicator_importance": ind_importance,
                        "henseler_normalized_weight": henseler_norm,
                        "performance": mean_rescaled,
                        "scale_min": lo,
                        "scale_max": hi,
                    }
                )

        rows: list[dict] = []
        for lv in path.index:
            if lv == target:
                continue
            if lv not in relevant.index:
                continue
            total = float(relevant.loc[lv, "total"])
            if total == 0:
                continue
            rows.append(
                {
                    "lv": lv,
                    "importance": total,
                    "performance": lv_perf.get(lv, float("nan")),
                }
            )
        lv_df = (
            pd.DataFrame(rows).set_index("lv")
            if rows
            else pd.DataFrame(columns=["importance", "performance"])
        )
        ind_df = pd.DataFrame(ind_rows)
        if not ind_df.empty:
            ind_df = ind_df.set_index(["lv", "indicator"])
        return lv_df, ind_df

    def latent_variables(self) -> pd.DataFrame:
        """LV-level importance/performance for the target.

        One row per LV with a non-zero total effect on the target.
        Columns:
          - ``importance``: standardized total effect on the target.
          - ``performance``: mean of the 0-100-rescaled LV score.
        """
        if self.__lv_table is None:
            self.__lv_table, self.__indicator_table = self.__compute()
        return self.__lv_table

    def indicators(self) -> pd.DataFrame:
        """Indicator-level performance and weight contribution.

        Indexed by ``(lv, indicator)``. Columns:
          - ``outer_weight``: raw weight from the outer model.
          - ``normalized_weight``: weight divided by the per-LV weight sum
            (legacy "share of weight" definition).
          - ``indicator_importance``: indicator's total effect on the IPMA
            target, computed as ``outer_weight × lv_importance`` per the
            Henseler IPMA convention (Ringle & Sarstedt 2016).
          - ``henseler_normalized_weight``: ``indicator_importance /
            lv_importance``. This is the value SmartPLS-style IPMA tables
            report under "Normalized Weight" and is what practitioners
            compare against the LV-level importance.
          - ``performance``: mean of the 0-100-rescaled indicator.
          - ``scale_min``, ``scale_max``: bounds used for rescaling.
        """
        if self.__indicator_table is None:
            self.__lv_table, self.__indicator_table = self.__compute()
        return self.__indicator_table
