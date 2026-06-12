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

"""Bootstrap-based Multi-Group Analysis with three pairwise difference tests.

Complements the permutation-based :class:`openpls.mga.MGA` with bootstrap
inference on the difference between two groups. For every path coefficient,
outer loading, outer weight, total effect, and specific / total indirect
effect, three tests are reported side by side:

1. **Henseler (2007) distribution-based** — empirical one- and two-tailed
   p-values from pairwise comparison of the two per-group bootstrap
   resampling distributions (Sarstedt, Henseler & Ringle 2011).
2. **Parametric test (Chin 2000)** — pooled-variance independent-samples
   t-test using per-group bootstrap standard errors.
3. **Welch-Satterthwaite test** — unequal-variance variant of (2).

The per-group bootstrap summary (``original``, ``mean``, ``std_error``,
``t_value``, ``p_value``) and per-group bias-corrected CIs are emitted
alongside the contrast for parity with SmartPLS-style reporting.

Use :class:`openpls.mga.MGA` for permutation-based testing (row-shuffling
across the full pooled sample) or when more than two groups are involved.
:class:`BootstrapMGA` is required for parametric / Welch parity and matches
the SmartPLS "Bootstrap MGA" output table.

References
----------
- Henseler, J. (2007). *A new and simple approach to multi-group analysis
  in partial least squares path modeling*. PLS'07.
- Chin, W. W. (2000). *Frequently Asked Questions — Partial Least Squares
  & PLS-Graph*. http://disc-nt.cba.uh.edu/chin/plsfaq.htm.
- Sarstedt, M., Henseler, J., & Ringle, C. M. (2011). *Multigroup analysis
  in PLS path modeling*. Advances in International Marketing, 22, 195-218.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022). *A
  Primer on PLS-SEM* (3rd ed.), §4.6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist

from openpls.config import Config
from openpls.long_bootstrap import LongBootstrap
from openpls.mga import GroupSpec, _row_mask
from openpls.scheme import Scheme
from openpls.specific_indirect import enumerate_chains

__all__ = ["BootstrapMGA", "GroupSpec"]


# ---------------------------------------------------------------------------
# Per-quantity helpers
# ---------------------------------------------------------------------------


def _henseler_p_one_tailed(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Empirical P(beta_a <= beta_b) from the two resampling distributions.

    Used as the "1-tailed (A vs B)" p-value: small p supports H1: beta_a > beta_b.
    Implemented with sort + searchsorted (O((m+n) log n)) instead of the naive
    O(mn) pairwise loop.
    """
    a = samples_a[np.isfinite(samples_a)]
    b = samples_b[np.isfinite(samples_b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    b_sorted = np.sort(b)
    # For each a[i]: count b's strictly greater than a[i].
    greater = b_sorted.size - np.searchsorted(b_sorted, a, side="right")
    return float(greater.sum()) / (a.size * b_sorted.size)


def _parametric_t(
    diff: float, se_a: float, se_b: float, n_a: int, n_b: int
) -> tuple[float, float, float]:
    """Chin (2000) pooled-variance t-test on the bootstrap-SE difference.

    Hair Primer 3rd ed. Eq. 4.7:
        t = diff /
            [ sqrt( (n_a-1)^2/(n_a+n_b-2) * SE_a^2
                  + (n_b-1)^2/(n_a+n_b-2) * SE_b^2 )
              * sqrt(1/n_a + 1/n_b) ]
        df = n_a + n_b - 2
    """
    if not (np.isfinite(diff) and np.isfinite(se_a) and np.isfinite(se_b)):
        return float("nan"), float("nan"), float("nan")
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    df = n_a + n_b - 2
    pooled_sq = (
        ((n_a - 1) ** 2) / df * (se_a ** 2)
        + ((n_b - 1) ** 2) / df * (se_b ** 2)
    )
    if pooled_sq <= 0:
        return float("nan"), float(df), float("nan")
    se_diff = float(np.sqrt(pooled_sq) * np.sqrt(1.0 / n_a + 1.0 / n_b))
    if se_diff <= 0:
        return float("nan"), float(df), float("nan")
    t = diff / se_diff
    p = float(2.0 * (1.0 - t_dist.cdf(abs(t), df)))
    return float(t), float(df), p


def _welch_t(
    diff: float, se_a: float, se_b: float, n_a: int, n_b: int
) -> tuple[float, float, float]:
    """Welch-Satterthwaite unequal-variance t-test on the bootstrap-SE difference.

    Hair Primer 3rd ed. Eq. 4.8:
        v_a = (n_a-1)/n_a * SE_a^2,   v_b = (n_b-1)/n_b * SE_b^2
        t   = diff / sqrt(v_a + v_b)
        df  = (v_a + v_b)^2 / ( v_a^2/(n_a-1) + v_b^2/(n_b-1) )
    """
    if not (np.isfinite(diff) and np.isfinite(se_a) and np.isfinite(se_b)):
        return float("nan"), float("nan"), float("nan")
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    v_a = ((n_a - 1) / n_a) * (se_a ** 2)
    v_b = ((n_b - 1) / n_b) * (se_b ** 2)
    s = v_a + v_b
    if s <= 0:
        return float("nan"), float("nan"), float("nan")
    t = diff / float(np.sqrt(s))
    denom = (v_a ** 2) / (n_a - 1) + (v_b ** 2) / (n_b - 1)
    df = float((s ** 2) / denom) if denom > 0 else float("nan")
    p = float(2.0 * (1.0 - t_dist.cdf(abs(t), df))) if np.isfinite(df) else float("nan")
    return float(t), df, p


def _bca_ci(
    samples: np.ndarray, point: float, alpha: float
) -> tuple[float, float]:
    """Bias-corrected percentile CI (Efron 1987). Falls back to plain
    percentiles when bias-correction is undefined."""
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return float("nan"), float("nan")
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    if not np.isfinite(point):
        return float(np.quantile(valid, lo_q)), float(np.quantile(valid, hi_q))
    p = float(np.mean(valid < point))
    if p <= 0 or p >= 1:
        return float(np.quantile(valid, lo_q)), float(np.quantile(valid, hi_q))
    z0 = float(norm.ppf(p))
    alpha_lo = float(np.clip(norm.cdf(z0 + (z0 + norm.ppf(lo_q))), 0.0, 1.0))
    alpha_hi = float(np.clip(norm.cdf(z0 + (z0 + norm.ppf(hi_q))), 0.0, 1.0))
    return float(np.quantile(valid, alpha_lo)), float(np.quantile(valid, alpha_hi))


@dataclass
class _GroupSummary:
    """Per-group raw bootstrap summary for one scalar quantity."""
    original: float
    mean: float
    se: float
    t_value: float
    p_value: float
    ci_lo: float
    ci_hi: float
    samples: np.ndarray


def _summarize(samples: np.ndarray, point: float, alpha: float) -> _GroupSummary:
    """Per-group inference on RAW (unflipped) bootstrap samples.

    SmartPLS' "no sign changes" setting requires raw resamples; sign-flipping
    would invalidate the cross-group Henseler comparison because each group's
    flip would be independent.
    """
    valid = samples[np.isfinite(samples)]
    if valid.size < 2:
        return _GroupSummary(
            original=point,
            mean=float("nan"),
            se=float("nan"),
            t_value=float("nan"),
            p_value=float("nan"),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            samples=valid,
        )
    mean = float(valid.mean())
    se = float(valid.std(ddof=1))
    t = float(point / se) if se > 0 and np.isfinite(point) else float("nan")
    # Per-group p-value uses the normal approximation (z-test) as SmartPLS does
    # in its per-group bootstrapping report.
    p = float(2.0 * (1.0 - norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")
    ci_lo, ci_hi = _bca_ci(valid, point, alpha)
    return _GroupSummary(
        original=float(point),
        mean=mean,
        se=se,
        t_value=t,
        p_value=p,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        samples=valid,
    )


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _row(
    id_pairs: list[tuple[str, str]],
    group_names: tuple[str, str],
    sum_a: _GroupSummary,
    sum_b: _GroupSummary,
    n_a: int,
    n_b: int,
) -> dict[str, Any]:
    """Construct one inference row combining the per-group summaries with the
    three difference tests. ``id_pairs`` is a list of (column_name, value)
    tuples for the identifier columns (e.g. ``[("source", "ATTR"),
    ("target", "COMP")]``)."""
    g_a, g_b = group_names
    diff = sum_a.original - sum_b.original
    p1 = _henseler_p_one_tailed(sum_a.samples, sum_b.samples)
    if np.isfinite(p1):
        p2 = float(2.0 * min(p1, 1.0 - p1))
    else:
        p2 = float("nan")
    t_param, df_param, pp_param = _parametric_t(diff, sum_a.se, sum_b.se, n_a, n_b)
    t_welch, df_welch, pp_welch = _welch_t(diff, sum_a.se, sum_b.se, n_a, n_b)
    row: dict[str, Any] = {k: v for k, v in id_pairs}
    row.update(
        {
            f"original_{g_a}": sum_a.original,
            f"original_{g_b}": sum_b.original,
            f"mean_{g_a}": sum_a.mean,
            f"mean_{g_b}": sum_b.mean,
            f"std_error_{g_a}": sum_a.se,
            f"std_error_{g_b}": sum_b.se,
            f"t_value_{g_a}": sum_a.t_value,
            f"t_value_{g_b}": sum_b.t_value,
            f"p_value_{g_a}": sum_a.p_value,
            f"p_value_{g_b}": sum_b.p_value,
            f"ci_bc_2_5_{g_a}": sum_a.ci_lo,
            f"ci_bc_97_5_{g_a}": sum_a.ci_hi,
            f"ci_bc_2_5_{g_b}": sum_b.ci_lo,
            f"ci_bc_97_5_{g_b}": sum_b.ci_hi,
            "difference": diff,
            "henseler_p_1tailed": p1,
            "henseler_p_2tailed": p2,
            "parametric_t": t_param,
            "parametric_df": df_param,
            "parametric_p": pp_param,
            "welch_t": t_welch,
            "welch_df": df_welch,
            "welch_p": pp_welch,
        }
    )
    return row


# ---------------------------------------------------------------------------
# BootstrapMGA class
# ---------------------------------------------------------------------------


class BootstrapMGA:
    """Bootstrap-based two-group MGA with Henseler / Parametric / Welch tests.

    Parameters
    ----------
    data, config, grouping_column, groups
        Same semantics as :class:`openpls.mga.MGA`. Exactly two groups must be
        provided; this matches the SmartPLS Bootstrap MGA reporting convention.
    scheme
        Inner weighting scheme (default :attr:`Scheme.CENTROID`).
    subsamples
        Number of bootstrap resamples per group (default 5000, matching the
        SmartPLS recommendation).
    seed
        RNG seed. Each group's :class:`LongBootstrap` uses ``seed + i`` to
        keep the two distributions independent yet reproducible.
    alpha
        Significance level for the per-group BCa confidence intervals.

    Each result accessor returns a wide DataFrame with one row per quantity
    and columns:

    - ``original_{group}``, ``mean_{group}``, ``std_error_{group}``,
      ``t_value_{group}``, ``p_value_{group}``,
      ``ci_bc_2_5_{group}``, ``ci_bc_97_5_{group}`` (one block per group)
    - ``difference`` (group_A - group_B)
    - ``henseler_p_1tailed``, ``henseler_p_2tailed``
    - ``parametric_t``, ``parametric_df``, ``parametric_p``
    - ``welch_t``, ``welch_df``, ``welch_p``
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        grouping_column: str,
        groups: list[GroupSpec],
        scheme: Scheme = Scheme.CENTROID,
        subsamples: int = 5000,
        seed: int | None = 42,
        alpha: float = 0.05,
    ):
        if grouping_column not in data.columns:
            raise ValueError(f"grouping_column {grouping_column!r} not in data")
        if len(groups) != 2:
            raise ValueError(
                "BootstrapMGA requires exactly 2 groups; use openpls.mga.MGA "
                "for permutation-based testing across more groups"
            )
        if subsamples < 2:
            raise ValueError("subsamples must be >= 2")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")

        self.__config = config
        self.__alpha = float(alpha)
        self.__subsamples = int(subsamples)

        series = data[grouping_column]
        subsets: list[tuple[GroupSpec, pd.DataFrame]] = []
        for g in groups:
            mask = _row_mask(series, g)
            sub = data.loc[mask].reset_index(drop=True)
            if len(sub) == 0:
                raise ValueError(f"group {g.name!r} matches zero rows")
            subsets.append((g, sub))

        self.__group_names = (subsets[0][0].name, subsets[1][0].name)
        self.__ns = (len(subsets[0][1]), len(subsets[1][1]))

        seed_base = 0 if seed is None else int(seed)
        self.__boot: list[LongBootstrap] = [
            LongBootstrap(
                sub,
                config,
                scheme,
                iterations=self.__subsamples,
                seed=seed_base + i,
                alpha=self.__alpha,
            )
            for i, (_, sub) in enumerate(subsets)
        ]
        self.__cache: dict[str, pd.DataFrame] = {}

    # ------- public accessors -------

    @property
    def group_names(self) -> tuple[str, str]:
        """Group labels in the order they were passed to the constructor."""
        return self.__group_names

    @property
    def group_sizes(self) -> tuple[int, int]:
        """Sample sizes per group."""
        return self.__ns

    def path_coefficients(self) -> pd.DataFrame:
        return self.__cache.setdefault("paths", self.__build_paths())

    def outer_loadings(self) -> pd.DataFrame:
        return self.__cache.setdefault(
            "loadings", self.__build_outer("loading")
        )

    def outer_weights(self) -> pd.DataFrame:
        return self.__cache.setdefault(
            "weights", self.__build_outer("weight")
        )

    def total_effects(self) -> pd.DataFrame:
        return self.__cache.setdefault("total_effects", self.__build_total_effects())

    def specific_indirect_effects(self) -> pd.DataFrame:
        if "sie" not in self.__cache:
            sie, tie = self.__build_indirect_effects()
            self.__cache["sie"] = sie
            self.__cache["tie"] = tie
        return self.__cache["sie"]

    def total_indirect_effects(self) -> pd.DataFrame:
        if "tie" not in self.__cache:
            sie, tie = self.__build_indirect_effects()
            self.__cache["sie"] = sie
            self.__cache["tie"] = tie
        return self.__cache["tie"]

    # ------- builders -------

    def __build_paths(self) -> pd.DataFrame:
        boot_a, boot_b = self.__boot
        keys = boot_a.path_keys
        # Sanity: both groups share the same Config, so path_keys must match.
        assert boot_a.path_keys == boot_b.path_keys, (
            "internal: path_keys differ between groups"
        )
        pt_a = boot_a.paths().set_index(["source", "target"])["original"]
        pt_b = boot_b.paths().set_index(["source", "target"])["original"]
        res_a = boot_a.resamples["pathCoefficients"]
        res_b = boot_b.resamples["pathCoefficients"]
        rows = []
        for k, (src, tgt) in enumerate(keys):
            sum_a = _summarize(res_a[:, k], float(pt_a.loc[(src, tgt)]), self.__alpha)
            sum_b = _summarize(res_b[:, k], float(pt_b.loc[(src, tgt)]), self.__alpha)
            rows.append(
                _row(
                    [("source", src), ("target", tgt)],
                    self.__group_names,
                    sum_a,
                    sum_b,
                    self.__ns[0],
                    self.__ns[1],
                )
            )
        return pd.DataFrame(rows)

    def __build_outer(self, column: str) -> pd.DataFrame:
        # ``column`` is either "loading" or "weight"; the index into resamples
        # is the same key list for both since LongBootstrap stores them in
        # parallel arrays.
        boot_a, boot_b = self.__boot
        keys = boot_a.outer_keys
        assert boot_a.outer_keys == boot_b.outer_keys, (
            "internal: outer_keys differ between groups"
        )
        if column == "loading":
            res_a = boot_a.resamples["outerLoadings"]
            res_b = boot_b.resamples["outerLoadings"]
            point_a = boot_a.loadings().set_index(["lv", "indicator"])["original"]
            point_b = boot_b.loadings().set_index(["lv", "indicator"])["original"]
        elif column == "weight":
            res_a = boot_a.resamples["outerWeights"]
            res_b = boot_b.resamples["outerWeights"]
            point_a = boot_a.weights().set_index(["lv", "indicator"])["original"]
            point_b = boot_b.weights().set_index(["lv", "indicator"])["original"]
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown outer column {column!r}")
        rows = []
        for k, (lv, ind) in enumerate(keys):
            sum_a = _summarize(
                res_a[:, k], float(point_a.loc[(lv, ind)]), self.__alpha
            )
            sum_b = _summarize(
                res_b[:, k], float(point_b.loc[(lv, ind)]), self.__alpha
            )
            rows.append(
                _row(
                    [("lv", lv), ("indicator", ind)],
                    self.__group_names,
                    sum_a,
                    sum_b,
                    self.__ns[0],
                    self.__ns[1],
                )
            )
        return pd.DataFrame(rows)

    def __build_total_effects(self) -> pd.DataFrame:
        boot_a, boot_b = self.__boot
        lv_names = boot_a.lv_names
        assert boot_a.lv_names == boot_b.lv_names, (
            "internal: lv_names differ between groups"
        )
        res_a = boot_a.resamples["totalEffects"]  # (iter, n, n)
        res_b = boot_b.resamples["totalEffects"]
        # Point estimates per group come from the existing total_effects()
        # DataFrame; rebuild a (source, target) -> original lookup.
        pt_a = boot_a.total_effects().set_index(["source", "target"])["original"]
        pt_b = boot_b.total_effects().set_index(["source", "target"])["original"]
        rows = []
        for i, tgt in enumerate(lv_names):
            for j, src in enumerate(lv_names):
                if i == j:
                    continue
                key = (src, tgt)
                if key not in pt_a.index and key not in pt_b.index:
                    continue
                point_a_val = float(pt_a.loc[key]) if key in pt_a.index else float("nan")
                point_b_val = float(pt_b.loc[key]) if key in pt_b.index else float("nan")
                sum_a = _summarize(res_a[:, i, j], point_a_val, self.__alpha)
                sum_b = _summarize(res_b[:, i, j], point_b_val, self.__alpha)
                rows.append(
                    _row(
                        [("source", src), ("target", tgt)],
                        self.__group_names,
                        sum_a,
                        sum_b,
                        self.__ns[0],
                        self.__ns[1],
                    )
                )
        return pd.DataFrame(rows)

    def __build_indirect_effects(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        boot_a, boot_b = self.__boot
        lv_names = boot_a.lv_names
        path_matrix = self.__config.path()
        edge_idx = {(src, tgt): k for k, (src, tgt) in enumerate(boot_a.path_keys)}
        res_paths_a = boot_a.resamples["pathCoefficients"]
        res_paths_b = boot_b.resamples["pathCoefficients"]
        # Per-group point estimates for paths (used to multiply along chains).
        pt_a = boot_a.paths().set_index(["source", "target"])["original"]
        pt_b = boot_b.paths().set_index(["source", "target"])["original"]

        sie_rows: list[dict[str, Any]] = []
        tie_rows: list[dict[str, Any]] = []

        for src in lv_names:
            for tgt in lv_names:
                if src == tgt:
                    continue
                try:
                    chains = enumerate_chains(path_matrix, src, tgt)
                except (KeyError, ValueError):
                    chains = []
                if not chains:
                    continue
                tie_point_a = 0.0
                tie_point_b = 0.0
                tie_sample_a = np.zeros(self.__subsamples)
                tie_sample_b = np.zeros(self.__subsamples)
                contributed = False
                for chain in chains:
                    edge_keys = list(zip(chain[:-1], chain[1:]))
                    if any(k not in edge_idx for k in edge_keys):
                        continue
                    cols = [edge_idx[k] for k in edge_keys]
                    sp_a = np.prod(res_paths_a[:, cols], axis=1)
                    sp_b = np.prod(res_paths_b[:, cols], axis=1)
                    pa, pb = 1.0, 1.0
                    for a, b in edge_keys:
                        pa *= float(pt_a.loc[(a, b)])
                        pb *= float(pt_b.loc[(a, b)])
                    via = " -> ".join(chain[1:-1])
                    sum_a = _summarize(sp_a, pa, self.__alpha)
                    sum_b = _summarize(sp_b, pb, self.__alpha)
                    sie_rows.append(
                        _row(
                            [("source", src), ("target", tgt), ("via", via)],
                            self.__group_names,
                            sum_a,
                            sum_b,
                            self.__ns[0],
                            self.__ns[1],
                        )
                    )
                    tie_sample_a = tie_sample_a + sp_a
                    tie_sample_b = tie_sample_b + sp_b
                    tie_point_a += pa
                    tie_point_b += pb
                    contributed = True
                if not contributed:
                    continue
                sum_a = _summarize(tie_sample_a, tie_point_a, self.__alpha)
                sum_b = _summarize(tie_sample_b, tie_point_b, self.__alpha)
                tie_rows.append(
                    _row(
                        [("source", src), ("target", tgt)],
                        self.__group_names,
                        sum_a,
                        sum_b,
                        self.__ns[0],
                        self.__ns[1],
                    )
                )
        sie = pd.DataFrame(sie_rows) if sie_rows else pd.DataFrame()
        tie = pd.DataFrame(tie_rows) if tie_rows else pd.DataFrame()
        return sie, tie
