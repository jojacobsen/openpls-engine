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

"""Long-running bootstrap with progress callbacks and BCa confidence intervals.

Complements upstream :class:`openpls.bootstrap.Bootstrap` (which favours
multiprocessing for short runs in a single Python process) with a serial,
progress-reporting variant suited to Cloud-Run-style workloads where:

- the run may take minutes to hours,
- the caller wants to stream progress (e.g. into a Firestore document),
- the failure/success ratio matters and must be exposed,
- richer per-path statistics are needed (sign-flipped samples, two-sided
  normal-approximation p-values, percentile or BCa confidence intervals).

The API takes the same ``(config, data, scheme)`` triple as :class:`openpls.mga.MGA`
and :class:`openpls.q_squared.QSquared` so all OpenPLS extensions share the same
construction pattern.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from openpls.config import Config
from openpls.scheme import Scheme
from openpls.specific_indirect import enumerate_chains


def _flip_signs(samples: np.ndarray, reference: float) -> np.ndarray:
    """Resolves PLS sign indeterminacy by flipping samples that disagree in sign
    with the point estimate (when the flip looks like a true sign reversal rather
    than a value close to zero)."""
    if np.isnan(reference) or reference == 0:
        return samples
    flips = np.abs(samples + reference) < np.abs(samples - reference)
    out = samples.copy()
    out[flips] = -out[flips]
    return out


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    """Plain two-sided percentile CI at level ``1 - alpha``."""
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def _bca_ci(
    samples: np.ndarray,
    point: float,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected percentile CI (Efron 1987). Falls back to plain
    percentiles if the bias-correction is undefined (all samples on one side
    of the point)."""
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2
    if np.isnan(point):
        return float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))
    p = float(np.mean(samples < point))
    if p <= 0 or p >= 1:
        return float(np.quantile(samples, lo_q)), float(np.quantile(samples, hi_q))
    z0 = norm.ppf(p)
    z_lo = norm.ppf(lo_q)
    z_hi = norm.ppf(hi_q)
    alpha_lo = float(np.clip(norm.cdf(z0 + (z0 + z_lo)), 0.0, 1.0))
    alpha_hi = float(np.clip(norm.cdf(z0 + (z0 + z_hi)), 0.0, 1.0))
    return (
        float(np.quantile(samples, alpha_lo)),
        float(np.quantile(samples, alpha_hi)),
    )


def _two_sided_bootstrap_p(samples: np.ndarray, point: float) -> float:
    """Two-sided p-value derived from the bootstrap distribution centred at zero.

    Uses the recentred distribution ``samples - mean(samples)`` so that the
    null hypothesis of zero effect is what the resamples are tested against
    (Davison & Hinkley 1997, §4.4)."""
    valid = samples[~np.isnan(samples)]
    if valid.size < 2 or np.isnan(point) or point == 0.0:
        return float("nan")
    recentred = valid - valid.mean()
    extreme = np.sum(np.abs(recentred) >= abs(point))
    # +1 / +1 smoothing keeps the p-value strictly in (0, 1] even when no
    # resample is as extreme as the point estimate.
    return float((extreme + 1) / (valid.size + 1))


def _aggregate(samples: np.ndarray, point: float, alpha: float) -> dict[str, float]:
    flipped = _flip_signs(samples, point)
    valid = flipped[~np.isnan(flipped)]
    if valid.size < 2:
        return {
            "boot_mean": float("nan"),
            "se": float("nan"),
            "t": float("nan"),
            "p_value": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_percentile_lower": float("nan"),
            "ci_percentile_upper": float("nan"),
            "ci_bc_lower": float("nan"),
            "ci_bc_upper": float("nan"),
            "p_value_bootstrap": float("nan"),
            "valid": int(valid.size),
        }
    mean = float(valid.mean())
    se = float(valid.std(ddof=1))
    t_val = float(point / se) if se > 0 and not np.isnan(point) else float("nan")
    p_val = float(2 * (1 - norm.cdf(abs(t_val)))) if not np.isnan(t_val) else float("nan")
    perc_lo, perc_hi = _percentile_ci(valid, alpha)
    bc_lo, bc_hi = _bca_ci(valid, point, alpha)
    p_boot = _two_sided_bootstrap_p(valid, point)
    return {
        "boot_mean": mean,
        "se": se,
        "t": t_val,
        "p_value": p_val,
        # Backwards-compatible ``ci_lower`` / ``ci_upper`` keep the BC bounds
        # so existing consumers see the same numbers.
        "ci_lower": bc_lo,
        "ci_upper": bc_hi,
        "ci_percentile_lower": perc_lo,
        "ci_percentile_upper": perc_hi,
        "ci_bc_lower": bc_lo,
        "ci_bc_upper": bc_hi,
        "p_value_bootstrap": p_boot,
        "valid": int(valid.size),
    }


_INFERENCE_RENAME = {
    "boot_mean": "mean",
    "se": "std_error",
    "t": "t_value",
    "p_value_bootstrap": "p_value",
    "ci_percentile_lower": "ci_percentile_2_5",
    "ci_percentile_upper": "ci_percentile_97_5",
    "ci_bc_lower": "ci_bc_2_5",
    "ci_bc_upper": "ci_bc_97_5",
}

_INFERENCE_COLUMNS = [
    "original",
    "mean",
    "std_error",
    "t_value",
    "p_value",
    "ci_percentile_2_5",
    "ci_percentile_97_5",
    "ci_bc_2_5",
    "ci_bc_97_5",
]


def _to_inference(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """Project an internal aggregator frame onto the public inference schema.

    Renames the verbose internal column names to the canonical inference names
    (``mean``, ``std_error``, ``t_value``, ``p_value``, ``ci_percentile_*``,
    ``ci_bc_*``) and orders columns as ``id_cols`` followed by the inference
    columns. Missing columns are filled with NaN so callers can rely on the
    schema regardless of the entity type.
    """
    if df.empty:
        empty_cols = id_cols + _INFERENCE_COLUMNS
        return pd.DataFrame(columns=empty_cols)
    # Drop the legacy normal-approx ``p_value`` so the bootstrap-based
    # ``p_value_bootstrap`` rename does not collide.
    projected = df.drop(columns=["p_value"], errors="ignore")
    renamed = projected.rename(columns=_INFERENCE_RENAME)
    cols = id_cols + _INFERENCE_COLUMNS
    for col in cols:
        if col not in renamed.columns:
            renamed[col] = float("nan")
    return renamed.loc[:, cols].reset_index(drop=True)


class LongBootstrap:
    """Single-process bootstrap with progress callback and BCa CIs.

    Use :class:`openpls.bootstrap.Bootstrap` for the upstream multiprocessing
    implementation. ``LongBootstrap`` is intended for serial, long-running
    workloads where progress reporting is more useful than wall-clock speed.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 5000,
        seed: int | None = 42,
        alpha: float = 0.05,
        on_progress: Callable[[int, int], None] | None = None,
        progress_every: int = 100,
        min_success_ratio: float = 0.1,
    ):
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if not (0 <= min_success_ratio <= 1):
            raise ValueError("min_success_ratio must be in [0, 1]")

        self.__config = config
        self.__scheme = scheme
        self.__alpha = alpha
        self.__iterations = iterations
        self.__rng = np.random.default_rng(seed)
        self.__data = data
        self.__on_progress = on_progress
        self.__progress_every = max(1, progress_every)
        self.__min_success_ratio = min_success_ratio

        self.__path_keys = self.__extract_paths(config)
        self.__lv_names = list(config.path().index)
        self.__outer_keys = self.__extract_outer_keys(config, data)

        self.__compute()

    @staticmethod
    def __extract_paths(config: Config) -> list[tuple[str, str]]:
        path = config.path()
        pairs: list[tuple[str, str]] = []
        for target in path.index:
            for source in path.columns:
                if path.loc[target, source] == 1:
                    pairs.append((source, target))
        return pairs

    @staticmethod
    def __extract_outer_keys(config: Config, data: pd.DataFrame) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for lv in config.path().index:
            for ind in config.mvs(lv):
                if ind in data.columns:
                    out.append((lv, ind))
        return out

    def __fit(self, df: pd.DataFrame):
        from openpls.plspm import Plspm  # local import to avoid circular dependency

        return Plspm(df, self.__config, self.__scheme)

    def __total_effects_matrix(self, path_df: pd.DataFrame) -> np.ndarray:
        names = self.__lv_names
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        B = np.zeros((n, n), dtype=float)
        for source, target in self.__path_keys:
            try:
                B[idx[target], idx[source]] = float(path_df.loc[target, source])
            except (KeyError, TypeError, ValueError):
                pass
        try:
            return np.linalg.inv(np.eye(n) - B) - np.eye(n)
        except np.linalg.LinAlgError:
            return np.zeros((n, n))

    def __compute(self):
        n = len(self.__data)
        # Point estimates from the full sample
        point_fit = self.__fit(self.__data)
        point_paths = point_fit.path_coefficients()
        point_outer = point_fit.outer_model()
        point_total = self.__total_effects_matrix(point_paths)

        path_keys = self.__path_keys
        outer_keys = self.__outer_keys
        n_lvs = len(self.__lv_names)

        boot_paths = np.full((self.__iterations, len(path_keys)), np.nan)
        boot_outer_loading = np.full((self.__iterations, len(outer_keys)), np.nan)
        boot_outer_weight = np.full((self.__iterations, len(outer_keys)), np.nan)
        boot_total = np.full((self.__iterations, n_lvs, n_lvs), np.nan)

        completed = 0
        failed = 0
        last_report = time.time()

        for it in range(self.__iterations):
            idx = self.__rng.integers(0, n, size=n)
            sample = self.__data.iloc[idx].reset_index(drop=True)
            try:
                fit = self.__fit(sample)
                pdf = fit.path_coefficients()
                outer = fit.outer_model()
                total = self.__total_effects_matrix(pdf)
            except Exception:
                failed += 1
                continue
            for k, (src, tgt) in enumerate(path_keys):
                try:
                    boot_paths[it, k] = float(pdf.loc[tgt, src])
                except (KeyError, TypeError, ValueError):
                    pass
            for k, (_, ind) in enumerate(outer_keys):
                try:
                    boot_outer_loading[it, k] = float(outer.loc[ind, "loading"])
                    boot_outer_weight[it, k] = float(outer.loc[ind, "weight"])
                except (KeyError, TypeError, ValueError):
                    pass
            boot_total[it] = total
            completed += 1
            if self.__on_progress and (
                (it + 1) % self.__progress_every == 0 or time.time() - last_report > 5
            ):
                self.__on_progress(it + 1, self.__iterations)
                last_report = time.time()

        if self.__on_progress:
            self.__on_progress(self.__iterations, self.__iterations)

        floor = max(2, int(self.__min_success_ratio * self.__iterations))
        if completed < floor:
            raise RuntimeError(
                f"Bootstrap failed: only {completed} of {self.__iterations} "
                f"resamples succeeded (min required: {floor})."
            )

        self.__completed = completed
        self.__failed = failed
        self.__boot_paths = boot_paths
        self.__boot_outer_loading = boot_outer_loading
        self.__boot_outer_weight = boot_outer_weight
        self.__boot_total = boot_total
        self.__point_paths = point_paths
        self.__point_total = point_total
        self.__paths_df = self.__aggregate_paths(point_paths, boot_paths)
        self.__loadings_df = self.__aggregate_outer(point_outer, boot_outer_loading, "loading")
        self.__weights_df = self.__aggregate_outer(point_outer, boot_outer_weight, "weight")
        self.__total_df = self.__aggregate_total(point_total, boot_total)
        self.__sie_df, self.__tie_df = self.__aggregate_indirect(
            point_paths, boot_paths, point_total, boot_total
        )

    def __aggregate_paths(self, point_paths: pd.DataFrame, samples: np.ndarray) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for k, (src, tgt) in enumerate(self.__path_keys):
            try:
                original = float(point_paths.loc[tgt, src])
            except (KeyError, TypeError, ValueError):
                original = float("nan")
            stats = _aggregate(samples[:, k], original, self.__alpha)
            rows.append({"source": src, "target": tgt, "original": original, **stats})
        return pd.DataFrame(rows)

    def __aggregate_outer(
        self, point_outer: pd.DataFrame, samples: np.ndarray, column: str
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for k, (lv, ind) in enumerate(self.__outer_keys):
            try:
                original = float(point_outer.loc[ind, column])
            except (KeyError, TypeError, ValueError):
                original = float("nan")
            stats = _aggregate(samples[:, k], original, self.__alpha)
            rows.append({"lv": lv, "indicator": ind, "original": original, **stats})
        return pd.DataFrame(rows)

    def __aggregate_total(self, point_total: np.ndarray, samples: np.ndarray) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i, tgt in enumerate(self.__lv_names):
            for j, src in enumerate(self.__lv_names):
                if i == j:
                    continue
                point_val = float(point_total[i, j])
                col = samples[:, i, j]
                if abs(point_val) < 1e-9 and np.nanmax(np.abs(col)) < 1e-9:
                    continue
                stats = _aggregate(col, point_val, self.__alpha)
                rows.append({"source": src, "target": tgt, "original": point_val, **stats})
        return pd.DataFrame(rows)

    def __aggregate_indirect(
        self,
        point_paths: pd.DataFrame,
        boot_paths: np.ndarray,
        point_total: np.ndarray,
        boot_total: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute specific indirect effects (one per chain) and total indirect
        effects (one per source/target pair) with full inference."""
        path_matrix = self.__config.path()
        edge_index = {(src, tgt): k for k, (src, tgt) in enumerate(self.__path_keys)}

        sie_rows: list[dict[str, Any]] = []
        tie_rows: list[dict[str, Any]] = []

        for src in self.__lv_names:
            for tgt in self.__lv_names:
                if src == tgt:
                    continue
                try:
                    chains = enumerate_chains(path_matrix, src, tgt)
                except (KeyError, ValueError):
                    chains = []
                if not chains:
                    continue

                chain_products = []
                for chain in chains:
                    edge_keys = list(zip(chain[:-1], chain[1:]))
                    if any(k not in edge_index for k in edge_keys):
                        continue
                    cols = [edge_index[k] for k in edge_keys]
                    sample_prod = np.prod(boot_paths[:, cols], axis=1)
                    point_prod = 1.0
                    for a, b in edge_keys:
                        point_prod *= float(point_paths.loc[b, a])
                    chain_products.append((chain, sample_prod, point_prod))
                    via = list(chain[1:-1])
                    stats = _aggregate(sample_prod, point_prod, self.__alpha)
                    sie_rows.append({
                        "source": src,
                        "target": tgt,
                        "via": " -> ".join(via),
                        "original": point_prod,
                        **stats,
                    })

                if not chain_products:
                    continue
                # Total indirect effect = sum of all specific indirect chain
                # products (Nitzl, Roldan & Cepeda 2016). Equivalent to
                # ``total - direct`` but computed directly so it covers cases
                # without a direct edge.
                tie_sample = np.zeros_like(chain_products[0][1])
                tie_point = 0.0
                for _, sample_prod, point_prod in chain_products:
                    tie_sample = tie_sample + sample_prod
                    tie_point += point_prod
                stats = _aggregate(tie_sample, tie_point, self.__alpha)
                tie_rows.append({
                    "source": src,
                    "target": tgt,
                    "original": tie_point,
                    **stats,
                })

        sie = pd.DataFrame(sie_rows) if sie_rows else pd.DataFrame(
            columns=["source", "target", "via", "original"]
        )
        tie = pd.DataFrame(tie_rows) if tie_rows else pd.DataFrame(
            columns=["source", "target", "original"]
        )
        return sie, tie

    @property
    def completed(self) -> int:
        """Number of resamples that fitted successfully."""
        return self.__completed

    @property
    def failed(self) -> int:
        """Number of resamples that raised during fitting."""
        return self.__failed

    @property
    def alpha(self) -> float:
        """Significance level used for confidence intervals."""
        return self.__alpha

    def paths(self) -> pd.DataFrame:
        """Per-path bootstrap statistics with BCa CI and normal-approx p-value."""
        return self.__paths_df

    def loadings(self) -> pd.DataFrame:
        """Per-indicator loading bootstrap statistics."""
        return self.__loadings_df

    def weights(self) -> pd.DataFrame:
        """Per-indicator weight bootstrap statistics."""
        return self.__weights_df

    def total_effects(self) -> pd.DataFrame:
        """Total-effect bootstrap statistics for every non-trivial LV pair."""
        return self.__total_df

    def specific_indirect_effects(self) -> pd.DataFrame:
        """Per-chain specific indirect effects with bootstrap inference.

        Each row is one ``source -> via... -> target`` chain (length >= 3 in
        the structural model). For models without any indirect chain the
        result is an empty DataFrame.
        """
        return self.__sie_df

    def total_indirect_effects(self) -> pd.DataFrame:
        """Total indirect effects (sum across chains) per source/target pair."""
        return self.__tie_df

    @property
    def inference(self) -> dict[str, pd.DataFrame]:
        """Unified bootstrap inference tables keyed by entity type.

        Returns a dict with six entries:

        - ``pathCoefficients``: direct structural paths.
        - ``outerLoadings``: indicator loadings.
        - ``outerWeights``: indicator weights.
        - ``specificIndirectEffects``: per-chain indirect effects.
        - ``totalIndirectEffects``: aggregate indirect effect per LV pair.
        - ``totalEffects``: direct + indirect per LV pair.

        Every DataFrame exposes the canonical inference columns:
        ``original``, ``mean``, ``std_error``, ``t_value``, ``p_value``,
        ``ci_percentile_2_5``, ``ci_percentile_97_5``, ``ci_bc_2_5``,
        ``ci_bc_97_5``. Identifier columns vary per entity (``source``/
        ``target`` for structural quantities, ``lv``/``indicator`` for outer
        quantities, with an extra ``via`` for specific indirect effects).
        """
        return {
            "pathCoefficients": _to_inference(self.__paths_df, ["source", "target"]),
            "outerLoadings": _to_inference(self.__loadings_df, ["lv", "indicator"]),
            "outerWeights": _to_inference(self.__weights_df, ["lv", "indicator"]),
            "specificIndirectEffects": _to_inference(
                self.__sie_df, ["source", "target", "via"]
            ),
            "totalIndirectEffects": _to_inference(self.__tie_df, ["source", "target"]),
            "totalEffects": _to_inference(self.__total_df, ["source", "target"]),
        }

    @property
    def resamples(self) -> dict[str, np.ndarray]:
        """Raw per-resample arrays kept for downstream analyses (e.g. MGA).

        Each entry is a NumPy array indexed by iteration along axis 0:

        - ``pathCoefficients``: shape ``(iterations, n_paths)`` aligned with
          :attr:`path_keys`.
        - ``outerLoadings`` / ``outerWeights``: shape ``(iterations,
          n_indicators)`` aligned with :attr:`outer_keys`.
        - ``totalEffects``: shape ``(iterations, n_lvs, n_lvs)`` aligned with
          :attr:`lv_names`.
        """
        return {
            "pathCoefficients": self.__boot_paths,
            "outerLoadings": self.__boot_outer_loading,
            "outerWeights": self.__boot_outer_weight,
            "totalEffects": self.__boot_total,
        }

    @property
    def path_keys(self) -> list[tuple[str, str]]:
        """Direct path identifiers (``source``, ``target``) used by :attr:`resamples`."""
        return list(self.__path_keys)

    @property
    def outer_keys(self) -> list[tuple[str, str]]:
        """Outer model identifiers (``lv``, ``indicator``) used by :attr:`resamples`."""
        return list(self.__outer_keys)

    @property
    def lv_names(self) -> list[str]:
        """Latent variable ordering used by :attr:`resamples` total effects."""
        return list(self.__lv_names)
