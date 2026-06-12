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

"""Measurement Invariance of Composite Models (MICOM).

Three-step procedure for testing whether composite constructs are measured
equivalently across two groups before drawing group-comparison conclusions
(e.g. MGA or moderation).

Step 1 — Configural invariance: identical indicators, identical data treatment
and identical algorithm settings for both groups. ``MICOM`` always satisfies
this by reusing one :class:`~openpls.config.Config` and one fit procedure for
all groups; the result is exposed verbatim in :meth:`MICOM.summary` for
audit purposes.

Step 2 — Compositional invariance: the correlation ``c`` between the
composite scores produced by group-A weights and group-B weights, evaluated
on the pooled covariance structure, must not be significantly below 1.
Tested with a permutation test on the group labels.

Step 3 — Equality of composite mean values and variances: once compositional
invariance holds, the means and variances of the composites must not differ
across groups. Tested by applying pooled-fit weights to each observation,
producing common-scale composite scores, then permuting the labels.

References
----------
- Henseler, J., Ringle, C. M., & Sarstedt, M. (2016). Testing measurement
  invariance of composites using partial least squares. International
  Marketing Review, 33(3), 405-431.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from openpls.config import Config
from openpls.mga import GroupSpec
from openpls.scheme import Scheme


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


@dataclass
class _Step2Stat:
    construct: str
    c: float


@dataclass
class _Step3Stat:
    construct: str
    mean_diff: float
    log_var_ratio: float
    var_diff: float


class MICOM:
    """Three-step measurement-invariance assessment for composite models.

    Restricted to **exactly two groups** by design: the canonical MICOM
    procedure is bivariate. For more than two groups, run MICOM pairwise.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        grouping_column: str,
        group_a: GroupSpec,
        group_b: GroupSpec,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 1000,
        seed: int | None = 42,
    ):
        if grouping_column not in data.columns:
            raise ValueError(f"grouping_column {grouping_column!r} not in data")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if group_a.name == group_b.name:
            raise ValueError("group_a and group_b must have distinct names")

        self.__config = config
        self.__scheme = scheme
        self.__iterations = iterations
        self.__rng = np.random.default_rng(seed)
        self.__group_a = group_a
        self.__group_b = group_b
        self.__alpha = 0.05

        series = data[grouping_column]
        mask_a = _row_mask(series, group_a)
        mask_b = _row_mask(series, group_b)
        if mask_a.sum() == 0:
            raise ValueError(f"group {group_a.name!r} matches zero rows")
        if mask_b.sum() == 0:
            raise ValueError(f"group {group_b.name!r} matches zero rows")
        if (mask_a & mask_b).any():
            raise ValueError(
                f"groups {group_a.name!r} and {group_b.name!r} overlap; "
                "MICOM requires disjoint groups"
            )

        df_a = data.loc[mask_a].reset_index(drop=True)
        df_b = data.loc[mask_b].reset_index(drop=True)
        pooled = pd.concat([df_a, df_b], ignore_index=True)

        self.__df_a = df_a
        self.__df_b = df_b
        self.__pooled = pooled
        self.__n_a = len(df_a)
        self.__n_total = len(pooled)

        self.__constructs = list(config.path().index)
        self.__indicators = {lv: list(config.mvs(lv)) for lv in self.__constructs}

        self.__weights_a = self.__fit_weights(df_a)
        self.__weights_b = self.__fit_weights(df_b)
        self.__weights_pooled = self.__fit_weights(pooled)

        # Aligned variants so that c-values are not flipped by PLS sign
        # indeterminacy (each per-group fit can land on +w or -w).
        self.__weights_b_aligned = {
            lv: self.__align_sign(self.__weights_a[lv], self.__weights_b[lv])
            for lv in self.__constructs
        }

        self.__observed_step2 = self.__compute_step2(
            self.__weights_a, self.__weights_b_aligned, pooled
        )
        self.__observed_step3 = self.__compute_step3(
            self.__weights_pooled, df_a, df_b, pooled
        )

        self.__step2_df: pd.DataFrame | None = None
        self.__step3_df: pd.DataFrame | None = None

    # ----- core helpers --------------------------------------------------

    def __fit_weights(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Returns ``{lv: weight_vector_over_its_indicators}`` for ``df``."""
        from openpls.plspm import Plspm  # local: avoid circular import

        fit = Plspm(df, self.__config, self.__scheme)
        outer = fit.outer_model()
        return {
            lv: outer.loc[self.__indicators[lv], "weight"].astype(float).copy()
            for lv in self.__constructs
        }

    @staticmethod
    def __align_sign(w_ref: pd.Series, w_other: pd.Series) -> pd.Series:
        """Flip ``w_other`` if it points in the opposite direction of ``w_ref``."""
        if float(np.dot(w_ref.values, w_other.values)) < 0:
            return -w_other
        return w_other

    @staticmethod
    def __c_from_weights(
        w_a: pd.Series, w_b: pd.Series, cov: pd.DataFrame
    ) -> float:
        """c = w_a' Σ w_b / sqrt((w_a' Σ w_a)(w_b' Σ w_b)) on the indicator block."""
        ind = list(w_a.index)
        sigma = cov.loc[ind, ind].values
        wa = w_a.values
        wb = w_b.values
        num = float(wa @ sigma @ wb)
        denom = float(np.sqrt((wa @ sigma @ wa) * (wb @ sigma @ wb)))
        if denom <= 0.0:
            return float("nan")
        return num / denom

    def __compute_step2(
        self,
        weights_a: dict[str, pd.Series],
        weights_b: dict[str, pd.Series],
        pooled: pd.DataFrame,
    ) -> dict[str, float]:
        all_indicators = sorted({m for inds in self.__indicators.values() for m in inds})
        cov = pooled[all_indicators].cov()
        return {
            lv: self.__c_from_weights(weights_a[lv], weights_b[lv], cov)
            for lv in self.__constructs
        }

    def __compute_step3(
        self,
        weights_pooled: dict[str, pd.Series],
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        pooled: pd.DataFrame,
    ) -> dict[str, tuple[float, float, float]]:
        """Composite mean diff, log-variance ratio, and raw variance diff."""
        all_indicators = sorted({m for inds in self.__indicators.values() for m in inds})
        mu = pooled[all_indicators].mean()
        sd = pooled[all_indicators].std(ddof=1).replace(0.0, np.nan)
        std_a = (df_a[all_indicators] - mu) / sd
        std_b = (df_b[all_indicators] - mu) / sd
        out: dict[str, tuple[float, float, float]] = {}
        for lv in self.__constructs:
            w = weights_pooled[lv]
            eta_a = std_a[w.index].values @ w.values
            eta_b = std_b[w.index].values @ w.values
            mean_diff = float(np.nanmean(eta_a) - np.nanmean(eta_b))
            var_a = float(np.nanvar(eta_a, ddof=1))
            var_b = float(np.nanvar(eta_b, ddof=1))
            var_diff = var_a - var_b
            if var_a <= 0.0 or var_b <= 0.0:
                log_ratio = float("nan")
            else:
                log_ratio = float(np.log(var_a / var_b))
            out[lv] = (mean_diff, log_ratio, var_diff)
        return out

    # ----- permutation tests --------------------------------------------

    def __permutation_step2(self) -> dict[str, float]:
        """One-sided lower-tail p-values: ``P(c_perm <= c_obs)``.

        Under H0 (compositional invariance), the permutation distribution of
        ``c`` clusters near 1. An observed ``c`` in the lower tail signals
        that the composites cannot be reconciled across groups.
        """
        observed = self.__observed_step2
        counts = {lv: 0 for lv in self.__constructs}
        valid = {lv: 0 for lv in self.__constructs}
        pooled = self.__pooled
        n_a = self.__n_a
        all_indicators = sorted({m for inds in self.__indicators.values() for m in inds})
        cov_full = pooled[all_indicators].cov()
        indices = np.arange(self.__n_total)
        for _ in range(self.__iterations):
            self.__rng.shuffle(indices)
            idx_a = indices[:n_a]
            idx_b = indices[n_a:]
            df_pa = pooled.iloc[idx_a].reset_index(drop=True)
            df_pb = pooled.iloc[idx_b].reset_index(drop=True)
            try:
                wa = self.__fit_weights(df_pa)
                wb = self.__fit_weights(df_pb)
            except Exception:
                continue
            for lv in self.__constructs:
                w_a = wa[lv]
                w_b = self.__align_sign(w_a, wb[lv])
                c_perm = self.__c_from_weights(w_a, w_b, cov_full)
                obs = observed[lv]
                if np.isnan(c_perm) or np.isnan(obs):
                    continue
                valid[lv] += 1
                if c_perm <= obs:
                    counts[lv] += 1
        return {
            lv: ((counts[lv] + 1) / (valid[lv] + 1)) if valid[lv] > 0 else float("nan")
            for lv in self.__constructs
        }

    def __permutation_step3(self) -> dict[str, tuple[float, float]]:
        """Two-sided permutation p-values for ``mean_diff`` and ``log_var_ratio``.

        Step 3 keeps the composite weights fixed at their pooled estimate, so
        each permutation is a label shuffle plus a linear combination — far
        cheaper than Step 2's refits.
        """
        all_indicators = sorted({m for inds in self.__indicators.values() for m in inds})
        pooled = self.__pooled
        mu = pooled[all_indicators].mean()
        sd = pooled[all_indicators].std(ddof=1).replace(0.0, np.nan)
        std_full = (pooled[all_indicators] - mu) / sd
        n_a = self.__n_a
        n_total = self.__n_total
        results: dict[str, tuple[float, float]] = {}
        composites = {
            lv: std_full[w.index].values @ w.values
            for lv, w in self.__weights_pooled.items()
        }
        indices = np.arange(n_total)
        # Counts: how many permutations produced |stat_perm| >= |stat_obs|.
        mean_counts = {lv: 0 for lv in self.__constructs}
        var_counts = {lv: 0 for lv in self.__constructs}
        mean_valid = {lv: 0 for lv in self.__constructs}
        var_valid = {lv: 0 for lv in self.__constructs}
        for _ in range(self.__iterations):
            self.__rng.shuffle(indices)
            idx_a = indices[:n_a]
            idx_b = indices[n_a:]
            for lv in self.__constructs:
                eta = composites[lv]
                eta_a = eta[idx_a]
                eta_b = eta[idx_b]
                obs_mean, obs_lvr, _ = self.__observed_step3[lv]
                # Mean
                if not np.isnan(obs_mean):
                    mean_valid[lv] += 1
                    perm_mean = float(np.nanmean(eta_a) - np.nanmean(eta_b))
                    if abs(perm_mean) >= abs(obs_mean):
                        mean_counts[lv] += 1
                # Variance
                if not np.isnan(obs_lvr):
                    var_a = float(np.nanvar(eta_a, ddof=1))
                    var_b = float(np.nanvar(eta_b, ddof=1))
                    if var_a > 0.0 and var_b > 0.0:
                        var_valid[lv] += 1
                        perm_lvr = float(np.log(var_a / var_b))
                        if abs(perm_lvr) >= abs(obs_lvr):
                            var_counts[lv] += 1
        for lv in self.__constructs:
            p_mean = (
                (mean_counts[lv] + 1) / (mean_valid[lv] + 1)
                if mean_valid[lv] > 0 else float("nan")
            )
            p_var = (
                (var_counts[lv] + 1) / (var_valid[lv] + 1)
                if var_valid[lv] > 0 else float("nan")
            )
            results[lv] = (p_mean, p_var)
        return results

    # ----- public API ----------------------------------------------------

    def step2(self) -> pd.DataFrame:
        """Compositional invariance (Step 2) per composite construct.

        Columns: ``construct``, ``c``, ``p_value``, ``compositional_invariance``.
        ``compositional_invariance`` is ``True`` when ``p_value >= alpha``
        (cannot reject the null that ``c = 1``).
        """
        if self.__step2_df is None:
            pvals = self.__permutation_step2()
            rows = []
            for lv in self.__constructs:
                p = pvals[lv]
                rows.append(
                    {
                        "construct": lv,
                        "c": self.__observed_step2[lv],
                        "p_value": p,
                        "compositional_invariance": bool(p >= self.__alpha)
                        if not np.isnan(p) else False,
                    }
                )
            self.__step2_df = pd.DataFrame(rows)
        return self.__step2_df

    def step3(self) -> pd.DataFrame:
        """Equality of composite means and variances (Step 3) per construct.

        Columns: ``construct, mean_diff, mean_p_value, mean_equal,
        log_var_ratio, var_diff, var_p_value, var_equal``.

        ``log_var_ratio`` (Henseler/Ringle/Sarstedt 2016 §3.4 convention) and
        ``var_diff = var_a - var_b`` (SmartPLS-style raw difference) carry the
        same sign and are both zero under H₀. Both are reported so that
        cross-implementation Δ comparisons in validation tables can pick the
        matching convention.
        """
        if self.__step3_df is None:
            pvals = self.__permutation_step3()
            rows = []
            for lv in self.__constructs:
                mean_diff, lvr, var_diff = self.__observed_step3[lv]
                p_mean, p_var = pvals[lv]
                rows.append(
                    {
                        "construct": lv,
                        "mean_diff": mean_diff,
                        "mean_p_value": p_mean,
                        "mean_equal": bool(p_mean >= self.__alpha)
                        if not np.isnan(p_mean) else False,
                        "log_var_ratio": lvr,
                        "var_diff": var_diff,
                        "var_p_value": p_var,
                        "var_equal": bool(p_var >= self.__alpha)
                        if not np.isnan(p_var) else False,
                    }
                )
            self.__step3_df = pd.DataFrame(rows)
        return self.__step3_df

    def summary(self) -> pd.DataFrame:
        """Per-construct verdict across all three MICOM steps.

        Columns: ``construct, c, compositional_invariance, mean_diff,
        mean_equal, log_var_ratio, var_equal, invariance``. The
        ``invariance`` column collapses the three steps:

        - ``"full"``: Step 2 passes AND both Step 3 sub-tests pass.
        - ``"partial"``: Step 2 passes but at least one Step 3 sub-test fails.
        - ``"none"``: Step 2 fails (composites are not comparable at all).
        """
        s2 = self.step2().set_index("construct")
        s3 = self.step3().set_index("construct")
        rows = []
        for lv in self.__constructs:
            ci = bool(s2.loc[lv, "compositional_invariance"])
            me = bool(s3.loc[lv, "mean_equal"])
            ve = bool(s3.loc[lv, "var_equal"])
            if not ci:
                level = "none"
            elif me and ve:
                level = "full"
            else:
                level = "partial"
            rows.append(
                {
                    "construct": lv,
                    "c": float(s2.loc[lv, "c"]),
                    "compositional_invariance": ci,
                    "mean_diff": float(s3.loc[lv, "mean_diff"]),
                    "mean_equal": me,
                    "log_var_ratio": float(s3.loc[lv, "log_var_ratio"]),
                    "var_equal": ve,
                    "invariance": level,
                }
            )
        return pd.DataFrame(rows)

    def group_sizes(self) -> dict[str, int]:
        """Observation counts per group (audit trail for Step 1)."""
        return {self.__group_a.name: self.__n_a, self.__group_b.name: self.__n_total - self.__n_a}


__all__ = ["MICOM", "GroupSpec"]
