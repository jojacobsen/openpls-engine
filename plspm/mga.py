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

"""Multi-Group Analysis (MGA) via Henseler's permutation test.

Refits the PLS model on each named subset of the data, reports per-group path
coefficients, and tests pairwise differences with a two-sided permutation test.

References
----------
- Henseler, J. (2007). A new and simple approach to multi-group analysis in
  PLS path modeling. PLS'07.
- Henseler, J., Ringle, C. M., & Sarstedt, M. (2016). Testing measurement
  invariance of composites using partial least squares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from plspm.config import Config
from plspm.scheme import Scheme


@dataclass
class GroupSpec:
    """One named subset of the data.

    Provide either ``values`` (categorical / list-membership) or ``range``
    (inclusive numeric interval; ``None`` means unbounded on that side).
    """
    name: str
    values: list[Any] | None = field(default=None)
    range: tuple[float | None, float | None] | None = field(default=None)


def _row_mask(series: pd.Series, group: GroupSpec) -> pd.Series:
    if group.range is not None:
        numeric = pd.to_numeric(series, errors="coerce")
        lo, hi = group.range
        mask = numeric.notna()
        if lo is not None:
            mask &= numeric >= lo
        if hi is not None:
            mask &= numeric <= hi
        return mask
    if group.values is not None and len(group.values) > 0:
        return series.isin(group.values)
    return pd.Series([False] * len(series), index=series.index)


class MGA:
    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        grouping_column: str,
        groups: list[GroupSpec],
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 5000,
        seed: int | None = 42,
    ):
        if grouping_column not in data.columns:
            raise ValueError(f"grouping_column {grouping_column!r} not in data")
        if len(groups) < 2:
            raise ValueError("MGA requires at least 2 groups")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        self.__config = config
        self.__scheme = scheme
        self.__groups = groups
        self.__iterations = iterations
        self.__rng = np.random.default_rng(seed)
        self.__paths = self.__extract_paths(config)

        series = data[grouping_column]
        self.__subsets: list[tuple[GroupSpec, pd.DataFrame]] = []
        for g in groups:
            mask = _row_mask(series, g)
            sub = data.loc[mask].reset_index(drop=True)
            if len(sub) == 0:
                raise ValueError(f"group {g.name!r} matches zero rows")
            self.__subsets.append((g, sub))

        self.__per_group_estimates: list[dict[tuple[str, str], float]] = [
            self.__fit_paths(sub) for _, sub in self.__subsets
        ]
        self.__comparisons_df: pd.DataFrame | None = None

    @staticmethod
    def __extract_paths(config: Config) -> list[tuple[str, str]]:
        """Returns (source, target) pairs from the path matrix (target rows, source cols)."""
        path = config.path()
        pairs: list[tuple[str, str]] = []
        for target in path.index:
            for source in path.columns:
                if path.loc[target, source] == 1:
                    pairs.append((source, target))
        return pairs

    def __fit_paths(self, df: pd.DataFrame) -> dict[tuple[str, str], float]:
        from plspm.plspm import Plspm  # local import to avoid circular dependency

        fit = Plspm(df, self.__config, self.__scheme)
        path_df = fit.path_coefficients()
        out: dict[tuple[str, str], float] = {}
        for source, target in self.__paths:
            try:
                out[(source, target)] = float(path_df.loc[target, source])
            except (KeyError, TypeError, ValueError):
                out[(source, target)] = np.nan
        return out

    def group_estimates(self) -> pd.DataFrame:
        """Per-group path estimates.

        Returns a DataFrame with columns ``("source", "target", "group", "n", "estimate")``.
        """
        rows = []
        for (g, sub), est in zip(self.__subsets, self.__per_group_estimates, strict=False):
            for (source, target), val in est.items():
                rows.append(
                    {
                        "group": g.name,
                        "n": int(len(sub)),
                        "source": source,
                        "target": target,
                        "estimate": val,
                    }
                )
        return pd.DataFrame(rows)

    def comparisons(self) -> pd.DataFrame:
        """Pairwise group comparisons with two-sided permutation p-values.

        Columns: ``groupA, groupB, source, target, estimateA, estimateB,
        difference, p_value``. Each pair is permuted ``iterations`` times.
        """
        if self.__comparisons_df is None:
            self.__comparisons_df = self.__compute_comparisons()
        return self.__comparisons_df

    def __compute_comparisons(self) -> pd.DataFrame:
        rows = []
        n_groups = len(self.__subsets)
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                g_a, df_a = self.__subsets[i]
                g_b, df_b = self.__subsets[j]
                est_a = self.__per_group_estimates[i]
                est_b = self.__per_group_estimates[j]
                observed = {
                    p: est_a[p] - est_b[p] for p in self.__paths
                }
                pvals = self.__permutation_pvalues(df_a, df_b, observed)
                for (source, target), diff in observed.items():
                    rows.append(
                        {
                            "groupA": g_a.name,
                            "groupB": g_b.name,
                            "source": source,
                            "target": target,
                            "estimateA": est_a[(source, target)],
                            "estimateB": est_b[(source, target)],
                            "difference": diff,
                            "p_value": pvals[(source, target)],
                        }
                    )
        return pd.DataFrame(rows)

    def __permutation_pvalues(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        observed: dict[tuple[str, str], float],
    ) -> dict[tuple[str, str], float]:
        n_a = len(df_a)
        pooled = pd.concat([df_a, df_b], ignore_index=True)
        n_total = len(pooled)
        counts = {k: 0 for k in observed}
        valid = {k: 0 for k in observed}
        indices = np.arange(n_total)
        for _ in range(self.__iterations):
            self.__rng.shuffle(indices)
            idx_a = indices[:n_a]
            idx_b = indices[n_a:]
            try:
                est_a = self.__fit_paths(pooled.iloc[idx_a].reset_index(drop=True))
                est_b = self.__fit_paths(pooled.iloc[idx_b].reset_index(drop=True))
            except Exception:
                continue
            for k, obs in observed.items():
                a = est_a.get(k, np.nan)
                b = est_b.get(k, np.nan)
                if np.isnan(obs) or np.isnan(a) or np.isnan(b):
                    continue
                valid[k] += 1
                if abs(a - b) >= abs(obs):
                    counts[k] += 1
        # Phipson & Smyth (2010) add-one smoothing to avoid p = 0.
        return {
            k: ((counts[k] + 1) / (valid[k] + 1)) if valid[k] > 0 else np.nan
            for k in observed
        }
