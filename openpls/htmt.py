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


class HTMT:
    """Heterotrait-Monotrait Ratio of Correlations (Henseler, Ringle & Sarstedt, 2015).

    For each pair ``(i, j)`` of latent variables::

        HTMT(i, j) = mean(|corr(x_i, x_j)|) / sqrt( mean_within(i) * mean_within(j) )

    where ``mean_within`` is the mean of *off-diagonal* absolute correlations
    among indicators of the same construct. Single-indicator constructs
    cannot have a within-block mean and are skipped (HTMT is NaN for pairs
    involving them).

    HTMT estimates the correlation between *reflectively measured* latent
    variables. Pairs that involve a formative (Mode B) construct are not in
    scope of the metric and are returned as ``NaN`` to avoid misleading
    discriminant-validity numbers.

    Henseler et al. (2015) suggest HTMT < 0.85 (conservative) or < 0.90
    (liberal) as the discriminant-validity threshold.
    """

    def __init__(self, config: c.Config, data: pd.DataFrame):
        lv_names = [lv for lv in config.path().columns]

        all_inds: list[str] = []
        seen: set[str] = set()
        lv_inds: dict[str, list[str]] = {}
        for lv in lv_names:
            inds: list[str] = []
            for mv in config.mvs(lv):
                if mv in data.columns:
                    inds.append(mv)
                    if mv not in seen:
                        seen.add(mv)
                        all_inds.append(mv)
            lv_inds[lv] = inds

        matrix = pd.DataFrame(np.nan, index=lv_names, columns=lv_names, dtype=float)
        for lv in lv_names:
            matrix.loc[lv, lv] = np.nan

        if len(lv_names) < 2 or not all_inds:
            self.__matrix = matrix
            return

        corr_abs = data[all_inds].corr().abs()

        def _mean_within(inds: list[str]) -> float | None:
            valid = [i for i in inds if i in corr_abs.index]
            if len(valid) < 2:
                return None
            sub = corr_abs.loc[valid, valid].to_numpy()
            n = sub.shape[0]
            # Off-diagonal upper triangle sum; off-diag count = n*(n-1)/2.
            return float(np.triu(sub, k=1).sum()) / (n * (n - 1) / 2.0)

        def _mean_between(a: list[str], b: list[str]) -> float | None:
            va = [i for i in a if i in corr_abs.index]
            vb = [i for i in b if i in corr_abs.index]
            if not va or not vb:
                return None
            return float(corr_abs.loc[va, vb].to_numpy().mean())

        within = {lv: _mean_within(lv_inds[lv]) for lv in lv_names}
        reflective = {lv for lv in lv_names if config.mode(lv) == Mode.A}

        for i, lv_a in enumerate(lv_names):
            for lv_b in lv_names[i + 1 :]:
                if lv_a not in reflective or lv_b not in reflective:
                    continue
                mw_a, mw_b = within[lv_a], within[lv_b]
                if mw_a is None or mw_b is None or mw_a <= 0 or mw_b <= 0:
                    continue
                mb = _mean_between(lv_inds[lv_a], lv_inds[lv_b])
                if mb is None:
                    continue
                ratio = mb / float(np.sqrt(mw_a * mw_b))
                matrix.loc[lv_a, lv_b] = ratio
                matrix.loc[lv_b, lv_a] = ratio

        self.__matrix = matrix

    def matrix(self) -> pd.DataFrame:
        """Full symmetric HTMT matrix indexed by latent variable name.

        Diagonal entries are ``NaN`` (HTMT is undefined within a construct).
        Pairs that involve a single-indicator construct are also ``NaN``.
        """
        return self.__matrix

    def pairs(self) -> pd.DataFrame:
        """Long-format view of the HTMT matrix — one row per unique LV pair.

        Useful for tabular display and CSV export. Columns: ``lv_a``,
        ``lv_b``, ``htmt``. Pairs with undefined HTMT (NaN) are omitted.
        """
        names = list(self.__matrix.index)
        rows = []
        for i, lv_a in enumerate(names):
            for lv_b in names[i + 1 :]:
                val = self.__matrix.loc[lv_a, lv_b]
                if pd.notna(val):
                    rows.append({"lv_a": lv_a, "lv_b": lv_b, "htmt": float(val)})
        return pd.DataFrame(rows, columns=["lv_a", "lv_b", "htmt"])
