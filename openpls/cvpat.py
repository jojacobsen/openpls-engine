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

"""Cross-Validated Predictive Ability Test (CVPAT, Liengaard et al. 2021).

CVPAT tests whether a PLS-SEM model has significantly better out-of-sample
predictive ability than a naive benchmark. For each observation in a k-fold
cross-validation, prediction errors are summed across the indicators of one
or more endogenous latent variables to produce a per-observation loss under
PLS-SEM and under the benchmark. A one-sided paired t-test on the loss
differences ``d_i = loss_i^PLS - loss_i^benchmark`` under
``H_0: E[d] >= 0`` rejects when PLS-SEM's predictive loss is significantly
lower than the benchmark's.

Two benchmarks are supported:

- ``"IA"`` — the training-fold indicator average. This is the naive
  Q²_predict baseline (predict every held-out observation with the
  train-fold column mean).
- ``"LM"`` — a linear regression of each indicator on the direct
  antecedent LVs' indicators, fitted on the training fold. Matches
  the PLSpredict LM benchmark.

References
----------
- Liengaard, B. D., Sharma, P. N., Hult, G. T. M., Jensen, M. B.,
  Sarstedt, M., Hair, J. F., & Ringle, C. M. (2021). Prediction:
  Coveted, yet forsaken? Introducing a cross-validated predictive
  ability test in partial least squares path modeling. Decision
  Sciences, 52(2), 362-392.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from openpls.config import Config
from openpls.scheme import Scheme

_VALID_BENCHMARKS = ("IA", "LM")


class CVPAT:
    """Cross-Validated Predictive Ability Test (Liengaard et al. 2021).

    Runs k-fold cross-validation over the sample and, for each observation,
    computes the sum of squared prediction errors over a set of endogenous
    LV indicators under (a) PLS-SEM and (b) a benchmark. The per-observation
    loss difference ``d_i`` feeds a one-sided paired t-test on
    ``H_0: E[d] >= 0`` (PLS-SEM is not better).

    Constructed via :meth:`~openpls.Plspm.cvpat`.

    Args:
        config: Model configuration.
        data: Full dataset (indicators only; will be reset-indexed
            internally).
        scheme: Inner weighting scheme used in each fold's PLS fit.
        benchmark: ``"IA"`` (indicator average, default) or ``"LM"``
            (direct-antecedents linear model).
        k: number of cross-validation folds (default 10).
        repeats: number of times to repeat the k-fold split with
            different permutations (default 1).
        seed: base RNG seed for fold permutations (default 42).
        alpha: significance level for the one-sided test (default 0.05).

    Raises:
        ValueError: if ``benchmark`` is not one of ``"IA"``, ``"LM"``;
            ``k`` < 2 or > n; ``repeats`` < 1; or ``alpha`` not in ``(0, 1)``.
    """

    def __init__(
        self,
        config: Config,
        data: pd.DataFrame,
        scheme: Scheme = Scheme.CENTROID,
        benchmark: str = "IA",
        k: int = 10,
        repeats: int = 1,
        seed: int | None = 42,
        alpha: float = 0.05,
    ):
        if benchmark not in _VALID_BENCHMARKS:
            raise ValueError(
                f"benchmark must be one of {_VALID_BENCHMARKS}, got {benchmark!r}"
            )
        if k < 2:
            raise ValueError("k must be >= 2")
        n = len(data)
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed sample size ({n})")
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.__config = config
        self.__data = data.reset_index(drop=True)
        self.__scheme = scheme
        self.__benchmark = benchmark
        self.__k = k
        self.__repeats = repeats
        self.__seed = seed
        self.__alpha = float(alpha)

        self.__losses: pd.DataFrame | None = None
        self.__overall: pd.Series | None = None
        self.__per_construct: pd.DataFrame | None = None

    def alpha(self) -> float:
        """Significance level used for the one-sided test."""
        return self.__alpha

    def benchmark(self) -> str:
        """Benchmark model name (``"IA"`` or ``"LM"``)."""
        return self.__benchmark

    def losses(self) -> pd.DataFrame:
        """Per-observation × per-indicator squared prediction errors.

        Long-format DataFrame with columns ``observation``, ``lv``,
        ``indicator``, ``squared_error_pls``, ``squared_error_benchmark``.
        Averaged across repeats when ``repeats > 1``.
        """
        if self.__losses is None:
            self.__compute()
        return self.__losses

    def overall(self) -> pd.Series:
        """Model-level CVPAT: sum losses across ALL endogenous indicators.

        Returns a Series with:

        - ``n``: number of observations contributing to the test.
        - ``mean_loss_pls``, ``mean_loss_benchmark``: mean per-observation
          summed squared error.
        - ``mean_difference``: ``mean(d_i) = mean_loss_pls -
          mean_loss_benchmark``. Negative → PLS-SEM has lower loss.
        - ``std_error``: standard error of ``mean(d_i)``.
        - ``t_statistic``, ``df``, ``p_one_sided``: paired one-sided
          t-test on ``H_0: E[d] >= 0``.
        - ``verdict``: ``"pls_better"`` (reject H_0 at alpha),
          ``"pls_worse"`` (mean_difference > 0 significantly),
          ``"no_difference"`` (fail to reject).
        """
        if self.__overall is None:
            self.__compute()
        return self.__overall

    def per_construct(self) -> pd.DataFrame:
        """Construct-level CVPAT: one test per endogenous LV.

        Sums losses across each LV's indicators separately, then runs the
        same one-sided paired t-test on per-observation loss differences.
        Same columns as :meth:`overall`, indexed by LV name.
        """
        if self.__per_construct is None:
            self.__compute()
        return self.__per_construct

    def __endogenous_indicators(self) -> list[tuple[str, str]]:
        path = self.__config.path()
        endo_lvs = [lv for lv in path.index if path.loc[lv].sum() > 0]
        return [
            (lv, ind)
            for lv in endo_lvs
            for ind in self.__config.mvs(lv)
            if ind in self.__data.columns
        ]

    def __lm_feature_cols(self, lv: str) -> list[str]:
        path = self.__config.path()
        direct = [p for p in path.columns if path.loc[lv, p] == 1]
        cols: list[str] = []
        for p in direct:
            for i in self.__config.mvs(p):
                if i in self.__data.columns and i not in cols:
                    cols.append(i)
        return cols

    def __topo_order(self) -> list[str]:
        path = self.__config.path()
        ordered: list[str] = []
        remaining = list(path.index)
        while remaining:
            progress = False
            for lv in list(remaining):
                preds = [
                    p for p in path.columns
                    if path.loc[lv, p] == 1 and p in remaining
                ]
                if not preds:
                    ordered.append(lv)
                    remaining.remove(lv)
                    progress = True
                    break
            if not progress:
                ordered.extend(remaining)
                break
        return ordered

    def __test_lv_scores(
        self,
        fit_outer: pd.DataFrame,
        fit_paths: pd.DataFrame,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        topo: list[str],
    ) -> dict[str, np.ndarray]:
        path = self.__config.path()
        scores: dict[str, np.ndarray] = {}
        for lv in topo:
            inds_lv = [i for i in self.__config.mvs(lv) if i in df_test.columns]
            preds_lv = [p for p in path.columns if path.loc[lv, p] == 1]
            if not preds_lv:
                block_std = np.zeros((len(df_test), len(inds_lv)), dtype=float)
                w_vec = np.zeros(len(inds_lv), dtype=float)
                for j, ind in enumerate(inds_lv):
                    train_col = df_train[ind].to_numpy(dtype=float)
                    train_mean = float(np.nanmean(train_col))
                    train_sd = float(np.nanstd(train_col, ddof=0))
                    test_col = df_test[ind].to_numpy(dtype=float)
                    if train_sd > 0:
                        block_std[:, j] = (test_col - train_mean) / train_sd
                    if ind in fit_outer.index:
                        w_vec[j] = float(fit_outer.loc[ind, "weight"])
                scores[lv] = block_std @ w_vec
            else:
                acc = np.zeros(len(df_test), dtype=float)
                for p in preds_lv:
                    if p not in scores:
                        continue
                    if lv in fit_paths.index and p in fit_paths.columns:
                        beta = float(fit_paths.loc[lv, p])
                    else:
                        beta = 0.0
                    acc = acc + beta * scores[p]
                scores[lv] = acc
        return scores

    def __compute(self) -> None:
        from openpls.plspm import Plspm  # local import to avoid circular dependency

        n = len(self.__data)
        endo_inds = self.__endogenous_indicators()
        topo = self.__topo_order()

        # (obs_id, lv, ind) → accumulated (sum_sq_pls, sum_sq_bench, count)
        pls_sq = {(obs, lv, ind): 0.0 for obs in range(n) for lv, ind in endo_inds}
        bench_sq = {(obs, lv, ind): 0.0 for obs in range(n) for lv, ind in endo_inds}
        counts = {(obs, lv, ind): 0 for obs in range(n) for lv, ind in endo_inds}

        for r in range(self.__repeats):
            seed = (self.__seed + r) if self.__seed is not None else None
            rng = np.random.default_rng(seed)
            order = rng.permutation(n)
            folds = np.array_split(order, self.__k)
            for test_idx in folds:
                if len(test_idx) == 0:
                    continue
                train_idx = np.setdiff1d(np.arange(n), test_idx)
                df_train = self.__data.iloc[train_idx].reset_index(drop=True)
                df_test = self.__data.iloc[test_idx].reset_index(drop=True)
                try:
                    fit = Plspm(df_train, self.__config, self.__scheme)
                except Exception:
                    continue
                outer = fit.outer_model()
                paths = fit.path_coefficients()
                test_scores = self.__test_lv_scores(
                    outer, paths, df_train, df_test, topo
                )

                for lv, ind in endo_inds:
                    if ind not in outer.index:
                        continue
                    try:
                        loading = float(outer.loc[ind, "loading"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    train_col = df_train[ind].to_numpy(dtype=float)
                    train_mean = float(np.nanmean(train_col))
                    train_sd = float(np.nanstd(train_col, ddof=0))
                    if not math.isfinite(train_mean) or train_sd == 0:
                        continue
                    actual = df_test[ind].to_numpy(dtype=float)
                    pls_pred = train_mean + train_sd * loading * test_scores[lv]

                    if self.__benchmark == "IA":
                        bench_pred = np.full(len(df_test), train_mean)
                    else:  # "LM"
                        feature_cols = self.__lm_feature_cols(lv)
                        if not feature_cols:
                            bench_pred = np.full(len(df_test), train_mean)
                        else:
                            X_train = df_train[feature_cols].to_numpy(dtype=float)
                            y_train = train_col
                            X_test = df_test[feature_cols].to_numpy(dtype=float)
                            train_valid = (
                                ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                            )
                            if train_valid.sum() <= len(feature_cols) + 1:
                                bench_pred = np.full(len(df_test), train_mean)
                            else:
                                X_train_c = sm.add_constant(
                                    X_train[train_valid], has_constant="add"
                                )
                                X_test_c = sm.add_constant(X_test, has_constant="add")
                                try:
                                    lm = sm.OLS(
                                        y_train[train_valid], X_train_c
                                    ).fit()
                                    bench_pred = X_test_c @ lm.params
                                except Exception:
                                    bench_pred = np.full(len(df_test), train_mean)

                    valid = (
                        ~np.isnan(actual)
                        & ~np.isnan(pls_pred)
                        & ~np.isnan(bench_pred)
                    )
                    if not valid.any():
                        continue
                    for j, obs in enumerate(test_idx):
                        if not valid[j]:
                            continue
                        pls_sq[(int(obs), lv, ind)] += (actual[j] - pls_pred[j]) ** 2
                        bench_sq[(int(obs), lv, ind)] += (
                            actual[j] - bench_pred[j]
                        ) ** 2
                        counts[(int(obs), lv, ind)] += 1

        # Long-format loss table averaged across repeats.
        loss_rows: list[dict] = []
        for (obs, lv, ind), c in counts.items():
            if c == 0:
                continue
            loss_rows.append({
                "observation": obs,
                "lv": lv,
                "indicator": ind,
                "squared_error_pls": pls_sq[(obs, lv, ind)] / c,
                "squared_error_benchmark": bench_sq[(obs, lv, ind)] / c,
            })
        self.__losses = pd.DataFrame(loss_rows)

        # Overall test: sum indicators across ALL endogenous LVs per observation.
        overall_pls = self.__losses.groupby("observation")["squared_error_pls"].sum()
        overall_bench = self.__losses.groupby("observation")[
            "squared_error_benchmark"
        ].sum()
        common_obs = overall_pls.index.intersection(overall_bench.index)
        d = (overall_pls.loc[common_obs] - overall_bench.loc[common_obs]).to_numpy()
        self.__overall = self.__test(d, overall_pls.loc[common_obs].mean(),
                                     overall_bench.loc[common_obs].mean())

        # Per-construct test.
        rows: list[dict] = []
        for lv in sorted({lv for lv, _ in endo_inds}):
            sub = self.__losses[self.__losses["lv"] == lv]
            pls_by_obs = sub.groupby("observation")["squared_error_pls"].sum()
            bench_by_obs = sub.groupby("observation")[
                "squared_error_benchmark"
            ].sum()
            common = pls_by_obs.index.intersection(bench_by_obs.index)
            d_lv = (pls_by_obs.loc[common] - bench_by_obs.loc[common]).to_numpy()
            row = self.__test(
                d_lv,
                pls_by_obs.loc[common].mean(),
                bench_by_obs.loc[common].mean(),
            )
            row["lv"] = lv
            rows.append(row.to_dict())
        self.__per_construct = pd.DataFrame(rows).set_index("lv")[
            [
                "n",
                "mean_loss_pls",
                "mean_loss_benchmark",
                "mean_difference",
                "std_error",
                "t_statistic",
                "df",
                "p_one_sided",
                "verdict",
            ]
        ]

    def __test(
        self, d: np.ndarray, mean_pls: float, mean_bench: float
    ) -> pd.Series:
        n = int(d.size)
        row = {
            "n": n,
            "mean_loss_pls": float(mean_pls),
            "mean_loss_benchmark": float(mean_bench),
            "mean_difference": float("nan"),
            "std_error": float("nan"),
            "t_statistic": float("nan"),
            "df": max(n - 1, 0),
            "p_one_sided": float("nan"),
            "verdict": "no_difference",
        }
        if n < 2:
            return pd.Series(row)
        mean_d = float(np.mean(d))
        sd_d = float(np.std(d, ddof=1))
        row["mean_difference"] = mean_d
        if sd_d == 0.0:
            # Zero variance → zero SE; t is ±∞ if mean_d != 0, else 0.
            row["std_error"] = 0.0
            if mean_d == 0.0:
                row["t_statistic"] = 0.0
                row["p_one_sided"] = 0.5
            else:
                row["t_statistic"] = -np.inf if mean_d < 0 else np.inf
                row["p_one_sided"] = 0.0 if mean_d < 0 else 1.0
        else:
            se = sd_d / math.sqrt(n)
            t = mean_d / se
            # H1: mean_d < 0 → left-tail p-value.
            p = float(stats.t.cdf(t, df=n - 1))
            row["std_error"] = se
            row["t_statistic"] = float(t)
            row["p_one_sided"] = p

        p = row["p_one_sided"]
        if not math.isfinite(p):
            row["verdict"] = "no_difference"
        elif mean_d < 0 and p < self.__alpha:
            row["verdict"] = "pls_better"
        elif mean_d > 0 and (1.0 - p) < self.__alpha:
            row["verdict"] = "pls_worse"
        else:
            row["verdict"] = "no_difference"
        return pd.Series(row)
