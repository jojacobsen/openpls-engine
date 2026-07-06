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

"""Common Method Bias diagnostic (Lindell & Whitney 2001) marker-variable test.

When all constructs in a survey are measured with the same instrument, the
observed correlations may be inflated by a shared response method (common
method variance, CMV). Lindell & Whitney (2001) propose a post-hoc
correction that uses a *marker variable* — an item theoretically
unrelated to any substantive construct — as a proxy for the CMV level.

Procedure
---------
1. Compute the correlations between the marker and each substantive
   latent variable. The smallest (by absolute value) is taken as the
   CMV estimate ``r_M``.
2. Partial-correlate every substantive pair by ``r_M``:

       r_adjusted = (r_unadjusted − r_M) / (1 − r_M).

3. Recompute the significance of the adjusted correlations using
   ``n − 3`` degrees of freedom (three variables involved).

If a correlation loses significance after adjustment, common method
variance is a plausible confound for that construct pair.

References
----------
- Lindell, M. K., & Whitney, D. J. (2001). Accounting for common method
  variance in cross-sectional research designs. Journal of Applied
  Psychology, 86(1), 114-121.
- Malhotra, N. K., Kim, S. S., & Patil, A. (2006). Common method variance
  in IS research: A comparison of alternative approaches and a reanalysis
  of past research. Management Science, 52(12), 1865-1883.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _significance(r: float, df: int, alpha: float) -> tuple[float, bool]:
    """Return the two-sided p-value and significance flag for a Pearson r."""
    if df <= 0 or not np.isfinite(r):
        return float("nan"), False
    if 1.0 - r * r <= 0.0:
        return 0.0, True
    t = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
    p = float(2.0 * stats.t.sf(abs(t), df))
    return p, bool(p < alpha)


class CMBLindellWhitney:
    """Lindell-Whitney (2001) marker-variable CMB adjustment.

    Constructed via :meth:`~openpls.Plspm.cmb_lindell_whitney`. Requires
    a marker variable — a manifest or latent variable that theory says
    should be uncorrelated with any substantive construct. The smallest
    ``|corr(marker, LV_i)|`` across constructs is taken as the CMV
    estimate ``r_M``; each substantive construct pair correlation is
    then partial-correlated by ``r_M`` and its two-sided p-value
    recomputed at ``n − 3`` degrees of freedom.

    Args:
        scores: DataFrame of substantive LV scores (one column per LV).
        marker: Series of marker values, aligned to ``scores`` by index.
            Rows with ``NaN`` in either the marker or a substantive LV
            are dropped from the pairwise correlation for that pair
            only (pairwise deletion).
        alpha: significance level (default ``0.05``).

    Raises:
        ValueError: if ``alpha`` is not in ``(0, 1)`` or if fewer than
            three usable observations remain after alignment.
    """

    def __init__(
        self,
        scores: pd.DataFrame,
        marker: pd.Series,
        alpha: float = 0.05,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        marker = marker.reindex(scores.index)
        if marker.dropna().shape[0] < 3:
            raise ValueError(
                "at least three non-NaN marker observations are required"
            )

        self.__scores = scores
        self.__marker = marker
        self.__alpha = float(alpha)
        self.__lvs = list(scores.columns)

        self.__marker_correlations = self.__compute_marker_correlations()
        # Lindell & Whitney (2001) use the smallest positive correlation
        # between the marker and the substantive constructs. In practice
        # implementations vary; we take the value with the smallest
        # absolute magnitude and preserve its sign, so downstream users
        # can see whether the CMV proxy is positive or negative.
        finite = self.__marker_correlations.dropna()
        if finite.empty:
            self.__r_m = float("nan")
        else:
            idx = finite.abs().idxmin()
            self.__r_m = float(finite.loc[idx])

        self.__unadjusted = scores.corr(method="pearson")
        self.__adjusted = self.__adjust_matrix(self.__unadjusted, self.__r_m)
        self.__table = self.__build_table()

    def __compute_marker_correlations(self) -> pd.Series:
        out: dict[str, float] = {}
        for lv in self.__lvs:
            paired = pd.concat([self.__scores[lv], self.__marker], axis=1).dropna()
            if len(paired) < 3:
                out[lv] = float("nan")
                continue
            out[lv] = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
        return pd.Series(out, name="marker_correlation")

    @staticmethod
    def __adjust_matrix(unadjusted: pd.DataFrame, r_m: float) -> pd.DataFrame:
        if not np.isfinite(r_m) or abs(1.0 - r_m) < 1e-12:
            return pd.DataFrame(
                np.full(unadjusted.shape, np.nan),
                index=unadjusted.index,
                columns=unadjusted.columns,
            )
        adjusted = (unadjusted - r_m) / (1.0 - r_m)
        # Diagonal remains 1.0 after adjustment by definition of a
        # marker-variable partial correlation.
        for i in adjusted.index:
            adjusted.loc[i, i] = 1.0
        return adjusted

    def __build_table(self) -> pd.DataFrame:
        n = int(self.__scores.dropna().shape[0])
        rows: list[dict] = []
        for i, lv_a in enumerate(self.__lvs):
            for lv_b in self.__lvs[i + 1 :]:
                r_u = float(self.__unadjusted.loc[lv_a, lv_b])
                r_a = float(self.__adjusted.loc[lv_a, lv_b])
                p_u, sig_u = _significance(r_u, n - 2, self.__alpha)
                p_a, sig_a = _significance(r_a, n - 3, self.__alpha)
                verdict = "unchanged"
                if sig_u and not sig_a:
                    verdict = "lost_significance"
                elif not sig_u and sig_a:
                    verdict = "gained_significance"
                rows.append({
                    "lv_a": lv_a,
                    "lv_b": lv_b,
                    "r_unadjusted": r_u,
                    "p_unadjusted": p_u,
                    "sig_unadjusted": sig_u,
                    "r_adjusted": r_a,
                    "p_adjusted": p_a,
                    "sig_adjusted": sig_a,
                    "verdict": verdict,
                })
        return pd.DataFrame(
            rows,
            index=[f"{r['lv_a']} <-> {r['lv_b']}" for r in rows],
        )

    def marker_correlations(self) -> pd.Series:
        """Correlation of the marker with each substantive latent variable.

        Indexed by LV name. Pairwise deletion within each pair; rows
        with NaN are dropped only for that pair.
        """
        return self.__marker_correlations

    def cmb_estimate(self) -> float:
        """Lindell-Whitney CMB proxy ``r_M``.

        The signed correlation with the smallest absolute magnitude
        among the marker-construct correlations. Small ``|r_M|``
        indicates minimal common method variance.
        """
        return self.__r_m

    def unadjusted(self) -> pd.DataFrame:
        """Uncorrected Pearson correlation matrix of the substantive LVs."""
        return self.__unadjusted

    def adjusted(self) -> pd.DataFrame:
        """CMB-adjusted correlation matrix.

        Off-diagonal ``r_A_ij = (r_U_ij − r_M) / (1 − r_M)``. Diagonal
        entries remain ``1.0``.
        """
        return self.__adjusted

    def table(self) -> pd.DataFrame:
        """Per-pair adjustment table.

        Indexed by ``"LV_a <-> LV_b"``. Columns:

        - ``lv_a``, ``lv_b``: latent-variable names.
        - ``r_unadjusted``, ``p_unadjusted``, ``sig_unadjusted``:
          uncorrected Pearson correlation, two-sided p-value at
          ``n − 2`` df, and significance flag.
        - ``r_adjusted``, ``p_adjusted``, ``sig_adjusted``: CMB-adjusted
          correlation, two-sided p-value at ``n − 3`` df, and
          significance flag.
        - ``verdict``: ``"lost_significance"`` if the pair was significant
          before adjustment but not after (CMV likely confounds this pair),
          ``"gained_significance"`` for the reverse, ``"unchanged"``
          otherwise.
        """
        return self.__table

    def alpha(self) -> float:
        """Significance level used for the two-sided tests."""
        return self.__alpha
