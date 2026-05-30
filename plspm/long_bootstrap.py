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

Complements upstream :class:`plspm.bootstrap.Bootstrap` (which favours
multiprocessing for short runs in a single Python process) with a serial,
progress-reporting variant suited to Cloud-Run-style workloads where:

- the run may take minutes to hours,
- the caller wants to stream progress (e.g. into a Firestore document),
- the failure/success ratio matters and must be exposed,
- richer per-path statistics are needed (sign-flipped samples, two-sided
  normal-approximation p-values, percentile or BCa confidence intervals).

The API takes the same ``(config, data, scheme)`` triple as :class:`plspm.mga.MGA`
and :class:`plspm.q_squared.QSquared` so all OpenPLS extensions share the same
construction pattern.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from plspm.config import Config
from plspm.scheme import Scheme


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


def _bca_ci(
    samples: np.ndarray,
    point: float,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected percentile CI. Falls back to plain percentiles if the
    bias-correction is undefined (all samples on one side of the point)."""
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
            "valid": int(valid.size),
        }
    mean = float(valid.mean())
    se = float(valid.std(ddof=1))
    t_val = float(point / se) if se > 0 and not np.isnan(point) else float("nan")
    p_val = float(2 * (1 - norm.cdf(abs(t_val)))) if not np.isnan(t_val) else float("nan")
    ci_lo, ci_hi = _bca_ci(valid, point, alpha)
    return {
        "boot_mean": mean,
        "se": se,
        "t": t_val,
        "p_value": p_val,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "valid": int(valid.size),
    }


class LongBootstrap:
    """Single-process bootstrap with progress callback and BCa CIs.

    Use :class:`plspm.bootstrap.Bootstrap` for the upstream multiprocessing
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
        from plspm.plspm import Plspm  # local import to avoid circular dependency

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
        self.__paths_df = self.__aggregate_paths(point_paths, boot_paths)
        self.__loadings_df = self.__aggregate_outer(point_outer, boot_outer_loading, "loading")
        self.__weights_df = self.__aggregate_outer(point_outer, boot_outer_weight, "weight")
        self.__total_df = self.__aggregate_total(point_total, boot_total)

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
