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

"""PLSpredict: out-of-sample predictive power via k-fold cross-validation.

For each indicator of an endogenous LV, predicts test-fold values from the
PLS model trained on the remaining folds and compares prediction error
against a linear-regression (LM) benchmark trained on the same predecessor
indicators.

Reports per-indicator RMSE, MAE, and MAPE (mean absolute percentage error
as a proportion, ``mean(|err / actual|)``, matching sklearn's convention)
for both PLS and LM, plus Q²_predict (``1 - SSE_pls / SSE_indicator_average``)
where the indicator average is the train-fold mean (Shmueli's naive
baseline). Each metric is reported in two variants:

- ``*_pls`` / ``*_lm`` — **out-of-sample** k-fold CV error.
- ``*_pls_in`` / ``*_lm_in`` — **in-sample** error from a single fit on
  all available rows (Shmueli et al. 2019, Table 6 panel A).

A negative Q²_predict means the PLS model does worse than just predicting
the training mean. PLS < LM RMSE on an indicator means PLS has predictive
advantage there; aggregating across indicators gives the standard
"high / medium / low / none predictive power" verdict.

References
----------
- Shmueli, G., Ray, S., Velasquez Estrada, J. M., & Chatla, S. B. (2016).
  The elephant in the room: Predictive performance of PLS models.
  Journal of Business Research, 69(10), 4552-4564.
- Shmueli, G., Sarstedt, M., Hair, J. F., Cheah, J.-H., Ting, H.,
  Vaithilingam, S., & Ringle, C. M. (2019). Predictive model assessment
  in PLS-SEM. European Journal of Marketing, 53(11), 2322-2347.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from openpls.config import Config
from openpls.scheme import Scheme


class PLSPredict:
    def __init__(
        self,
        config: Config,
        data: pd.DataFrame,
        scheme: Scheme = Scheme.CENTROID,
        k: int = 10,
        repeats: int = 1,
        seed: int | None = 42,
    ):
        if k < 2:
            raise ValueError("k must be >= 2")
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        n = len(data)
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed sample size ({n})")
        self.__config = config
        self.__data = data.reset_index(drop=True)
        self.__scheme = scheme
        self.__k = k
        self.__repeats = repeats
        self.__seed = seed
        self.__metrics: pd.DataFrame | None = None

    @property
    def k(self) -> int:
        return self.__k

    @property
    def repeats(self) -> int:
        return self.__repeats

    def metrics(self) -> pd.DataFrame:
        """Per-indicator prediction metrics.

        Indexed by ``(lv, indicator)``. Columns:

        Out-of-sample (k-fold CV):
          - ``rmse_pls``, ``mae_pls``, ``mape_pls``: PLS prediction error.
          - ``rmse_lm``, ``mae_lm``, ``mape_lm``: LM benchmark prediction
            error.
          - ``q2_predict``: ``1 - SSE_pls / SSE_indicator_average`` (the
            naive train-mean baseline).

        In-sample (single fit on the full data):
          - ``rmse_pls_in``, ``mae_pls_in``, ``mape_pls_in``: PLS error.
          - ``rmse_lm_in``, ``mae_lm_in``, ``mape_lm_in``: LM error.

        MAPE is the proportion ``mean(|err / actual|)``; rows where the
        actual value is zero are excluded from MAPE only (the other
        metrics still see them).
        """
        if self.__metrics is None:
            self.__metrics = self.__compute()
        return self.__metrics

    def summary(self) -> pd.Series:
        """Aggregate predictive-power verdict.

        Counts indicators where PLS RMSE is lower than LM RMSE. Returns
        per-indicator status: ``"better"`` (PLS < LM), ``"worse"``, or
        ``"tie"``. The proportion of ``better`` maps to the standard
        verdict (Shmueli et al. 2019, Table 6):

          - all indicators better → high predictive power
          - majority better       → medium
          - minority better       → low
          - none better           → no predictive power
        """
        m = self.metrics()
        diff = m["rmse_pls"] - m["rmse_lm"]
        out = pd.Series("tie", index=m.index, name="pls_vs_lm")
        out[diff < -1e-12] = "better"
        out[diff > 1e-12] = "worse"
        return out

    def __endogenous_indicators(self) -> list[tuple[str, str]]:
        path = self.__config.path()
        endo_lvs = [lv for lv in path.index if path.loc[lv].sum() > 0]
        return [
            (lv, ind)
            for lv in endo_lvs
            for ind in self.__config.mvs(lv)
            if ind in self.__data.columns
        ]

    def __topo_order(self) -> list[str]:
        path = self.__config.path()
        ordered: list[str] = []
        remaining = list(path.index)
        while remaining:
            progress = False
            for lv in list(remaining):
                preds = [p for p in path.columns if path.loc[lv, p] == 1 and p in remaining]
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

    def __compute_in_sample(
        self,
        endo_inds: list[tuple[str, str]],
        path: pd.DataFrame,
    ) -> pd.DataFrame:
        """In-sample PLS and LM errors from a single fit on the full data."""
        from openpls.plspm import Plspm  # local import to avoid circular dependency

        rows: list[dict] = []
        try:
            fit = Plspm(self.__data, self.__config, self.__scheme)
        except Exception:
            for lv, ind in endo_inds:
                rows.append({
                    "lv": lv, "indicator": ind,
                    "rmse_pls_in": float("nan"), "mae_pls_in": float("nan"),
                    "mape_pls_in": float("nan"),
                    "rmse_lm_in": float("nan"), "mae_lm_in": float("nan"),
                    "mape_lm_in": float("nan"),
                })
            return pd.DataFrame(rows).set_index(["lv", "indicator"])

        outer = fit.outer_model()
        scores = fit.scores()

        for lv, ind in endo_inds:
            row = {
                "lv": lv, "indicator": ind,
                "rmse_pls_in": float("nan"), "mae_pls_in": float("nan"),
                "mape_pls_in": float("nan"),
                "rmse_lm_in": float("nan"), "mae_lm_in": float("nan"),
                "mape_lm_in": float("nan"),
            }
            if ind not in outer.index:
                rows.append(row)
                continue
            try:
                loading = float(outer.loc[ind, "loading"])
            except (KeyError, TypeError, ValueError):
                rows.append(row)
                continue
            col = self.__data[ind].to_numpy(dtype=float)
            mean = float(np.nanmean(col))
            sd = float(np.nanstd(col, ddof=0))
            if not math.isfinite(mean) or sd == 0:
                rows.append(row)
                continue

            actual = col
            lv_score = scores.loc[:, lv].to_numpy(dtype=float)
            pls_pred = mean + sd * loading * lv_score

            preds_lv = [p for p in path.columns if path.loc[lv, p] == 1]
            feature_cols: list[str] = []
            for p in preds_lv:
                for i in self.__config.mvs(p):
                    if i in self.__data.columns and i not in feature_cols:
                        feature_cols.append(i)
            if not feature_cols:
                lm_pred = np.full(len(self.__data), mean)
            else:
                X = self.__data[feature_cols].to_numpy(dtype=float)
                y = col
                valid_train = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
                if valid_train.sum() <= len(feature_cols) + 1:
                    lm_pred = np.full(len(self.__data), mean)
                else:
                    X_c = sm.add_constant(X[valid_train], has_constant="add")
                    try:
                        lm = sm.OLS(y[valid_train], X_c).fit()
                        X_all = sm.add_constant(X, has_constant="add")
                        lm_pred = X_all @ lm.params
                    except Exception:
                        lm_pred = np.full(len(self.__data), mean)

            valid = ~np.isnan(actual) & ~np.isnan(pls_pred) & ~np.isnan(lm_pred)
            if not valid.any():
                rows.append(row)
                continue
            actual_v = actual[valid]
            pls_err = actual_v - pls_pred[valid]
            lm_err = actual_v - lm_pred[valid]
            n = int(valid.sum())
            row["rmse_pls_in"] = math.sqrt(float(np.sum(pls_err**2)) / n)
            row["mae_pls_in"] = float(np.sum(np.abs(pls_err))) / n
            row["rmse_lm_in"] = math.sqrt(float(np.sum(lm_err**2)) / n)
            row["mae_lm_in"] = float(np.sum(np.abs(lm_err))) / n
            nonzero = np.abs(actual_v) > 1e-12
            if nonzero.any():
                row["mape_pls_in"] = float(
                    np.mean(np.abs(pls_err[nonzero]) / np.abs(actual_v[nonzero]))
                )
                row["mape_lm_in"] = float(
                    np.mean(np.abs(lm_err[nonzero]) / np.abs(actual_v[nonzero]))
                )
            rows.append(row)

        return pd.DataFrame(rows).set_index(["lv", "indicator"])

    def __compute(self) -> pd.DataFrame:
        from openpls.plspm import Plspm  # local import to avoid circular dependency

        n = len(self.__data)
        endo_inds = self.__endogenous_indicators()
        path = self.__config.path()
        topo = self.__topo_order()

        pls_sq = {ind: 0.0 for _, ind in endo_inds}
        pls_abs = {ind: 0.0 for _, ind in endo_inds}
        lm_sq = {ind: 0.0 for _, ind in endo_inds}
        lm_abs = {ind: 0.0 for _, ind in endo_inds}
        ia_sq = {ind: 0.0 for _, ind in endo_inds}
        counts = {ind: 0 for _, ind in endo_inds}
        # MAPE accumulators (rows with actual == 0 are excluded here only).
        pls_pct = {ind: 0.0 for _, ind in endo_inds}
        lm_pct = {ind: 0.0 for _, ind in endo_inds}
        pct_counts = {ind: 0 for _, ind in endo_inds}

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
                test_scores = self.__test_lv_scores(outer, paths, df_train, df_test, topo)

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

                    preds_lv = [p for p in path.columns if path.loc[lv, p] == 1]
                    feature_cols: list[str] = []
                    for p in preds_lv:
                        for i in self.__config.mvs(p):
                            if i in df_train.columns and i not in feature_cols:
                                feature_cols.append(i)
                    if not feature_cols:
                        lm_pred = np.full(len(df_test), train_mean)
                    else:
                        X_train = df_train[feature_cols].to_numpy(dtype=float)
                        y_train = train_col
                        X_test = df_test[feature_cols].to_numpy(dtype=float)
                        train_valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                        if train_valid.sum() <= len(feature_cols) + 1:
                            lm_pred = np.full(len(df_test), train_mean)
                        else:
                            X_train_c = sm.add_constant(
                                X_train[train_valid], has_constant="add"
                            )
                            X_test_c = sm.add_constant(X_test, has_constant="add")
                            try:
                                lm = sm.OLS(y_train[train_valid], X_train_c).fit()
                                lm_pred = X_test_c @ lm.params
                            except Exception:
                                lm_pred = np.full(len(df_test), train_mean)

                    valid = ~np.isnan(actual) & ~np.isnan(pls_pred) & ~np.isnan(lm_pred)
                    if not valid.any():
                        continue
                    actual_v = actual[valid]
                    pls_err = actual_v - pls_pred[valid]
                    lm_err = actual_v - lm_pred[valid]
                    ia_err = actual_v - train_mean
                    pls_sq[ind] += float(np.sum(pls_err**2))
                    pls_abs[ind] += float(np.sum(np.abs(pls_err)))
                    lm_sq[ind] += float(np.sum(lm_err**2))
                    lm_abs[ind] += float(np.sum(np.abs(lm_err)))
                    ia_sq[ind] += float(np.sum(ia_err**2))
                    counts[ind] += int(valid.sum())
                    nonzero = np.abs(actual_v) > 1e-12
                    if nonzero.any():
                        pls_pct[ind] += float(np.sum(np.abs(pls_err[nonzero]) / np.abs(actual_v[nonzero])))
                        lm_pct[ind] += float(np.sum(np.abs(lm_err[nonzero]) / np.abs(actual_v[nonzero])))
                        pct_counts[ind] += int(nonzero.sum())

        rows: list[dict] = []
        for lv, ind in endo_inds:
            c = counts[ind]
            mape_c = pct_counts[ind]
            if c == 0:
                rows.append(
                    {
                        "lv": lv,
                        "indicator": ind,
                        "rmse_pls": float("nan"),
                        "mae_pls": float("nan"),
                        "mape_pls": float("nan"),
                        "q2_predict": float("nan"),
                        "rmse_lm": float("nan"),
                        "mae_lm": float("nan"),
                        "mape_lm": float("nan"),
                    }
                )
                continue
            rmse_pls = math.sqrt(pls_sq[ind] / c)
            mae_pls = pls_abs[ind] / c
            rmse_lm = math.sqrt(lm_sq[ind] / c)
            mae_lm = lm_abs[ind] / c
            mape_pls = (pls_pct[ind] / mape_c) if mape_c > 0 else float("nan")
            mape_lm = (lm_pct[ind] / mape_c) if mape_c > 0 else float("nan")
            q2 = (
                1.0 - pls_sq[ind] / ia_sq[ind]
                if ia_sq[ind] > 0
                else float("nan")
            )
            rows.append(
                {
                    "lv": lv,
                    "indicator": ind,
                    "rmse_pls": rmse_pls,
                    "mae_pls": mae_pls,
                    "mape_pls": mape_pls,
                    "q2_predict": q2,
                    "rmse_lm": rmse_lm,
                    "mae_lm": mae_lm,
                    "mape_lm": mape_lm,
                }
            )
        out = pd.DataFrame(rows).set_index(["lv", "indicator"])
        in_sample = self.__compute_in_sample(endo_inds, path)
        return out.join(in_sample, how="left")
