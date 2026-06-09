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


def _gmean(values: np.ndarray) -> float | None:
    """Geometric mean of strictly positive values.

    Implemented as ``exp(mean(log(x)))`` for numerical stability. Returns
    ``None`` if the input is empty or contains any non-positive entry
    (the geometric mean is mathematically undefined or zero when any
    factor is zero, and HTMT2's denominator cannot meaningfully use such
    a value).
    """
    if values.size == 0:
        return None
    if not np.all(values > 0):
        return None
    return float(np.exp(np.mean(np.log(values))))


class HTMT2:
    """HTMT2: geometric-mean refinement of the Heterotrait-Monotrait Ratio
    of Correlations (Roemer, Schuberth & Henseler, 2021).

    Replaces the two arithmetic means in the original Henseler, Ringle &
    Sarstedt (2015) HTMT with geometric means. For each pair ``(i, j)``
    of latent variables::

        HTMT2(i, j) = gmean(|corr(x_i, x_j)|) /
                      sqrt(gmean_within(i) * gmean_within(j))

    where ``gmean_within`` is the geometric mean of *off-diagonal*
    absolute correlations among indicators of the same construct.

    The motivation (Roemer et al. 2021): the original HTMT can be
    biased when the loadings within a block are unequal — the
    arithmetic mean over-weights the strongest indicators. HTMT2
    weighs indicators uniformly on the log-scale and is consistent
    under the tau-equivalent / congeneric measurement model.

    Single-indicator constructs and pairs whose correlations include
    any zero (geometric mean is undefined / zero) are skipped (NaN in
    the matrix, omitted from ``pairs()``). The same conservative
    discriminant-validity thresholds as HTMT apply (HTMT2 < 0.85 / 0.90).
    """

    def __init__(self, config: c.Config, data: pd.DataFrame):
        lv_names = list(config.path().columns)

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

        if len(lv_names) < 2 or not all_inds:
            self.__matrix = matrix
            return

        corr_abs = data[all_inds].corr().abs()

        def _gmean_within(inds: list[str]) -> float | None:
            valid = [i for i in inds if i in corr_abs.index]
            if len(valid) < 2:
                return None
            sub = corr_abs.loc[valid, valid].to_numpy()
            # Off-diagonal upper-triangle entries.
            triu = sub[np.triu_indices_from(sub, k=1)]
            return _gmean(triu)

        def _gmean_between(a: list[str], b: list[str]) -> float | None:
            va = [i for i in a if i in corr_abs.index]
            vb = [i for i in b if i in corr_abs.index]
            if not va or not vb:
                return None
            block = corr_abs.loc[va, vb].to_numpy().ravel()
            return _gmean(block)

        within = {lv: _gmean_within(lv_inds[lv]) for lv in lv_names}

        for i, lv_a in enumerate(lv_names):
            for lv_b in lv_names[i + 1 :]:
                gw_a, gw_b = within[lv_a], within[lv_b]
                if gw_a is None or gw_b is None or gw_a <= 0 or gw_b <= 0:
                    continue
                gb = _gmean_between(lv_inds[lv_a], lv_inds[lv_b])
                if gb is None:
                    continue
                ratio = gb / float(np.sqrt(gw_a * gw_b))
                matrix.loc[lv_a, lv_b] = ratio
                matrix.loc[lv_b, lv_a] = ratio

        self.__matrix = matrix

    def matrix(self) -> pd.DataFrame:
        """Full symmetric HTMT2 matrix indexed by latent variable name.

        Diagonal entries are ``NaN``. Pairs involving a single-indicator
        construct or any zero indicator correlation are also ``NaN``.
        """
        return self.__matrix

    def pairs(self) -> pd.DataFrame:
        """Long-format view — one row per unique LV pair with columns
        ``lv_a``, ``lv_b``, ``htmt2``. Undefined pairs are omitted.
        """
        names = list(self.__matrix.index)
        rows = []
        for i, lv_a in enumerate(names):
            for lv_b in names[i + 1 :]:
                val = self.__matrix.loc[lv_a, lv_b]
                if pd.notna(val):
                    rows.append({"lv_a": lv_a, "lv_b": lv_b, "htmt2": float(val)})
        return pd.DataFrame(rows, columns=["lv_a", "lv_b", "htmt2"])
