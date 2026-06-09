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

from itertools import combinations

import numpy as np
import pandas as pd

import openpls.config as c
from openpls.mode import Mode


def _canonical_tetrad(cov: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Canonical tetrad ``s_ij * s_kl - s_ik * s_jl`` for the index quadruple.

    One canonical tetrad is selected per indicator 4-tuple, giving
    ``C(p, 4)`` non-redundant tetrads per block. For a one-factor (purely
    reflective) measurement model all such tetrads have expectation zero.
    """
    return float(cov[i, j] * cov[k, l] - cov[i, k] * cov[j, l])


class CTAPLS:
    """Confirmatory Tetrad Analysis for PLS (Gudergan, Ringle, Wende & Will 2008).

    Diagnostic for the *outer model*: tests whether reflective (Mode A)
    specification of each block with four or more indicators is consistent
    with the data. The procedure relies on Bollen and Ting's (1993)
    vanishing-tetrad theorem — for a one-factor block every model-implied
    tetrad of the indicator covariance matrix has expectation zero. Sample
    tetrads are bootstrapped to obtain two-sided p-values for the null
    ``H0: tau = 0``; the Holm step-down correction is applied within each
    block to control the family-wise error rate at ``alpha``. If any tetrad
    in a block is rejected, the reflective specification is not supported
    and a formative (Mode B) specification should be considered.

    Only Mode A blocks with at least four indicators are tested. Mode B
    blocks and reflective blocks with fewer than four indicators are
    omitted (tetrads are undefined / vacuously satisfied).
    """

    def __init__(
        self,
        config: c.Config,
        data: pd.DataFrame,
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: int | None = 42,
    ):
        if n_boot < 50:
            raise ValueError(f"n_boot must be >= 50, got {n_boot}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.__alpha = alpha
        self.__n_boot = n_boot
        self.__tetrads, self.__summary = self.__compute(
            config, data, n_boot, alpha, seed
        )

    @staticmethod
    def __compute(
        config: c.Config,
        data: pd.DataFrame,
        n_boot: int,
        alpha: float,
        seed: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        tetrad_rows: list[dict] = []
        summary_rows: list[dict] = []
        for lv in config.path().columns:
            if config.mode(lv) != Mode.A:
                continue
            inds = [mv for mv in config.mvs(lv) if mv in data.columns]
            if len(inds) < 4:
                continue
            block = data[inds].dropna()
            if block.shape[0] < 10:
                continue
            block_np = block.to_numpy(dtype=float)
            n_obs, p = block_np.shape
            quads = list(combinations(range(p), 4))
            m = len(quads)
            sample_cov = np.cov(block_np, rowvar=False, ddof=1)
            observed = np.array(
                [_canonical_tetrad(sample_cov, *quad) for quad in quads]
            )
            boot_values = np.empty((n_boot, m), dtype=float)
            for b in range(n_boot):
                idx = rng.integers(n_obs, size=n_obs)
                cov_b = np.cov(block_np[idx, :], rowvar=False, ddof=1)
                for t, quad in enumerate(quads):
                    boot_values[b, t] = _canonical_tetrad(cov_b, *quad)
            ses = boot_values.std(axis=0, ddof=1)
            # Two-sided percentile p-value under H0: tau = 0. Center the
            # bootstrap distribution on zero before comparing magnitudes.
            centered = boot_values - boot_values.mean(axis=0, keepdims=True)
            p_values = np.mean(
                np.abs(centered) >= np.abs(observed), axis=0
            )
            # Floor at 1/n_boot so log scales and Holm comparisons stay sane.
            p_values = np.maximum(p_values, 1.0 / n_boot)
            decisions = CTAPLS.__holm(p_values, alpha)
            n_rejected = int(sum(d == "reject" for d in decisions))
            block_decision = (
                "reflective rejected"
                if n_rejected > 0
                else "reflective supported"
            )
            for t, quad in enumerate(quads):
                i, j, k, l = quad
                tetrad_rows.append(
                    {
                        "lv": lv,
                        "indicators": f"{inds[i]},{inds[j]},{inds[k]},{inds[l]}",
                        "tetrad": float(observed[t]),
                        "boot_se": float(ses[t]),
                        "p_value": float(p_values[t]),
                        "holm_decision": decisions[t],
                    }
                )
            summary_rows.append(
                {
                    "lv": lv,
                    "n_indicators": len(inds),
                    "n_tetrads": m,
                    "n_rejected": n_rejected,
                    "decision": block_decision,
                }
            )
        tetrads_df = pd.DataFrame(
            tetrad_rows,
            columns=[
                "lv",
                "indicators",
                "tetrad",
                "boot_se",
                "p_value",
                "holm_decision",
            ],
        )
        summary_df = pd.DataFrame(
            summary_rows,
            columns=[
                "lv",
                "n_indicators",
                "n_tetrads",
                "n_rejected",
                "decision",
            ],
        )
        return tetrads_df, summary_df

    @staticmethod
    def __holm(p_values: np.ndarray, alpha: float) -> list[str]:
        """Holm step-down rejection given a vector of p-values."""
        m = len(p_values)
        order = np.argsort(p_values)
        decisions = ["fail to reject"] * m
        for rank, t in enumerate(order):
            if p_values[t] < alpha / (m - rank):
                decisions[t] = "reject"
            else:
                break
        return decisions

    def tetrads(self) -> pd.DataFrame:
        """Per-tetrad results.

        Long format with one row per tested tetrad and columns ``lv``,
        ``indicators`` (the four indicator names that form the tetrad,
        comma-separated), ``tetrad`` (observed sample value), ``boot_se``,
        ``p_value``, and ``holm_decision`` (``"reject"`` /
        ``"fail to reject"``, after the within-block Holm correction at the
        configured ``alpha``).
        """
        return self.__tetrads

    def summary(self) -> pd.DataFrame:
        """Per-block CTA-PLS verdict.

        One row per tested Mode A block (≥ 4 indicators) with columns
        ``lv``, ``n_indicators``, ``n_tetrads``, ``n_rejected``, and
        ``decision`` (``"reflective supported"`` when no tetrad rejects
        after Holm correction; ``"reflective rejected"`` otherwise).
        """
        return self.__summary

    def alpha(self) -> float:
        """Family-wise significance level used for the Holm correction."""
        return self.__alpha

    def n_boot(self) -> int:
        """Number of bootstrap resamples used to estimate tetrad SEs."""
        return self.__n_boot
