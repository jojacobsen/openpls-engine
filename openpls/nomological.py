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

"""Nomological validity: directional hypothesis testing on LV correlations.

Nomological validity is established when correlations between latent
variables move in the direction that substantive theory predicts. Each
hypothesis specifies a construct pair and an expected sign (``"+"`` or
``"-"``); the test evaluates whether the observed Pearson correlation
between the two LVs (a) has the expected sign and (b) is significant at
the chosen α level via a one-sided t-test with ``n - 2`` degrees of
freedom.

The verdict is ``"supported"`` when the sign matches AND the one-sided
p-value is below α, and ``"not supported"`` otherwise. A construct
network passes the nomological validity check when every theoretically
justified hypothesis is supported.

References
----------
- Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in
  psychological tests. Psychological Bulletin, 52(4), 281-302.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022).
  A primer on partial least squares structural equation modeling
  (PLS-SEM) (3rd ed.). Sage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

Hypothesis = tuple[str, str, str]


def _validate_sign(sign: str) -> str:
    if sign not in ("+", "-"):
        raise ValueError(f"expected sign must be '+' or '-', got {sign!r}")
    return sign


class NomologicalValidity:
    """Directional hypothesis testing on latent-variable correlations.

    Constructed via :meth:`~openpls.Plspm.nomological_validity`. Each
    hypothesis is a triple ``(source, target, expected_sign)`` where
    ``expected_sign`` is ``"+"`` or ``"-"``. Bivariate Pearson
    correlations between the two LV score columns are evaluated with a
    one-sided t-test:

    - ``t = r * sqrt(n - 2) / sqrt(1 - r²)``
    - the sign matches when ``sign(r) == expected_sign``
    - ``supported`` when the sign matches AND ``p_one_sided < alpha``

    Args:
        scores: DataFrame of LV scores (one column per latent variable).
        hypotheses: sequence of ``(source, target, sign)`` triples. Both
            LV names must exist as columns of ``scores``. Sign must be
            ``"+"`` or ``"-"``. Duplicate hypotheses are preserved (the
            output row order matches the input order).
        alpha: significance level for the one-sided test (default 0.05).

    Raises:
        KeyError: if a hypothesis references an unknown LV name.
        ValueError: if a sign is not ``"+"`` / ``"-"``, or if ``source``
            equals ``target``, or if ``alpha`` is not in ``(0, 1)``.
    """

    def __init__(
        self,
        scores: pd.DataFrame,
        hypotheses: Sequence[Hypothesis],
        alpha: float = 0.05,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        for source, target, sign in hypotheses:
            _validate_sign(sign)
            if source == target:
                raise ValueError(
                    f"hypothesis source and target must differ ({source!r})"
                )
            if source not in scores.columns:
                raise KeyError(f"LV {source!r} not found in scores")
            if target not in scores.columns:
                raise KeyError(f"LV {target!r} not found in scores")

        self.__scores = scores
        self.__hypotheses = list(hypotheses)
        self.__alpha = float(alpha)
        self.__correlations = scores.corr(method="pearson")
        self.__table = self.__compute_table()

    def __compute_table(self) -> pd.DataFrame:
        n = len(self.__scores)
        rows: list[dict] = []
        for source, target, expected_sign in self.__hypotheses:
            r = float(self.__correlations.loc[source, target])
            if not np.isfinite(r):
                rows.append({
                    "from": source,
                    "to": target,
                    "expected_sign": expected_sign,
                    "correlation": np.nan,
                    "t_statistic": np.nan,
                    "df": max(n - 2, 0),
                    "p_one_sided": np.nan,
                    "sign_matches": False,
                    "verdict": "not supported",
                })
                continue

            df = n - 2
            if df <= 0:
                t_stat = np.nan
                p_one_sided = np.nan
            elif 1.0 - r * r <= 0.0:
                # Perfect correlation: t → ±∞, one-sided p in the correct
                # tail collapses to 0 or 1 depending on the sign match.
                t_stat = np.inf if r > 0.0 else -np.inf
                if expected_sign == "+":
                    p_one_sided = 0.0 if r > 0.0 else 1.0
                else:
                    p_one_sided = 0.0 if r < 0.0 else 1.0
            else:
                t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
                # One-sided p-value in the hypothesised direction:
                # "+" hypothesis rejects when t is large positive;
                # "-" hypothesis rejects when t is large negative.
                if expected_sign == "+":
                    p_one_sided = float(stats.t.sf(t_stat, df))
                else:
                    p_one_sided = float(stats.t.cdf(t_stat, df))

            sign_matches = (
                (expected_sign == "+" and r > 0.0)
                or (expected_sign == "-" and r < 0.0)
            )
            supported = bool(
                sign_matches
                and np.isfinite(p_one_sided)
                and p_one_sided < self.__alpha
            )
            rows.append({
                "from": source,
                "to": target,
                "expected_sign": expected_sign,
                "correlation": r,
                "t_statistic": float(t_stat),
                "df": df,
                "p_one_sided": p_one_sided,
                "sign_matches": sign_matches,
                "verdict": "supported" if supported else "not supported",
            })

        return pd.DataFrame(
            rows,
            index=[f"{r['from']} -> {r['to']}" for r in rows],
        )

    def correlations(self) -> pd.DataFrame:
        """Pearson correlation matrix over all latent variables in ``scores``.

        Symmetric ``k × k`` matrix (``k`` = number of LVs). Cell ``[i, j]``
        is the Pearson correlation of the LV scores between ``LV_i`` and
        ``LV_j``. Diagonal entries are ``1.0``.
        """
        return self.__correlations

    def table(self) -> pd.DataFrame:
        """Per-hypothesis verdict table.

        Indexed by ``"source -> target"``. Columns:

        - ``from``, ``to``: LV names.
        - ``expected_sign``: ``"+"`` or ``"-"``.
        - ``correlation``: Pearson r between the two LV score columns.
        - ``t_statistic``: ``r * sqrt(n - 2) / sqrt(1 - r²)``.
        - ``df``: degrees of freedom (``n - 2``).
        - ``p_one_sided``: one-sided p-value in the hypothesised direction.
        - ``sign_matches``: whether the observed sign of ``r`` matches the
          hypothesised sign.
        - ``verdict``: ``"supported"`` iff both the sign matches AND
          ``p_one_sided < alpha``; otherwise ``"not supported"``.
        """
        return self.__table

    def alpha(self) -> float:
        """Significance level used for the one-sided tests."""
        return self.__alpha
