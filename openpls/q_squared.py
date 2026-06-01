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

"""Stone-Geisser Q² (cross-validated redundancy) via blindfolding.

For each endogenous latent variable, every D-th row of its indicators is
replaced by the column mean of the non-omitted rows; the PLS model is re-fit;
the omitted indicator values are predicted from the inner model and the
remaining indicators are predicted from `mean + loading * sd * lv_score`.

Q² = 1 − SSE/SSO, where SSE is squared prediction error on omitted cells and
SSO is squared deviation from the training mean. Q² > 0 indicates predictive
relevance (Stone 1974; Geisser 1974; Henseler & Sarstedt 2013).
"""

import numpy as np
import pandas as pd

from openpls.config import Config
from openpls.scheme import Scheme


class QSquared:
    def __init__(
        self,
        config: Config,
        data: pd.DataFrame,
        scheme: Scheme,
        omission_distance: int = 7,
    ):
        self.__config = config
        self.__data = data
        self.__scheme = scheme
        if omission_distance < 2:
            raise ValueError("omission_distance must be >= 2")
        self.__omission_distance = omission_distance
        self.__values: pd.DataFrame | None = None

    def values(self) -> pd.DataFrame:
        """Returns a DataFrame indexed by endogenous LV with a `q_squared` column."""
        if self.__values is None:
            self.__values = self.__compute()
        return self.__values

    @property
    def omission_distance(self) -> int:
        return self.__omission_distance

    def __endogenous(self) -> list[str]:
        path = self.__config.path()
        return [lv for lv in path.index if path.loc[lv].sum() > 0]

    def __compute(self) -> pd.DataFrame:
        from openpls.plspm import Plspm  # local import to avoid circular dependency

        df = self.__data
        endo = self.__endogenous()
        rows: list[dict] = []
        for target in endo:
            inds = [i for i in self.__config.mvs(target) if i in df.columns]
            if not inds:
                rows.append({"lv": target, "q_squared": np.nan})
                continue
            sse = 0.0
            sso = 0.0
            for d in range(self.__omission_distance):
                row_mask = np.arange(len(df)) % self.__omission_distance == d
                if not row_mask.any() or row_mask.all():
                    continue
                df_in = df.copy()
                # pandas 2.2+ rejects float assignment into int columns;
                # promote the target indicator columns to float before imputing.
                df_in[inds] = df_in[inds].astype(float)
                for ind in inds:
                    col_mean = df.loc[~row_mask, ind].mean()
                    df_in.loc[row_mask, ind] = col_mean
                try:
                    fit = Plspm(df_in, self.__config, self.__scheme)
                except Exception:
                    continue
                scores = fit.scores()
                outer = fit.outer_model()
                path_df = fit.path_coefficients()
                if target not in scores.columns:
                    continue

                predecessors = [
                    lv for lv in self.__config.path().index
                    if self.__config.path().loc[target, lv] == 1
                ]
                contributions = pd.Series(0.0, index=scores.index, dtype=float)
                for src in predecessors:
                    if src in scores.columns and target in path_df.index and src in path_df.columns:
                        beta = float(path_df.loc[target, src])
                        contributions = contributions.add(beta * scores[src], fill_value=0.0)
                score_series = pd.Series(np.nan, index=df.index, dtype=float)
                score_series.loc[contributions.index] = contributions.values
                score_col = score_series.to_numpy(dtype=float)

                for ind in inds:
                    if ind not in outer.index:
                        continue
                    try:
                        loading = float(outer.loc[ind, "loading"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    actual = df[ind].to_numpy(dtype=float)
                    train = ~row_mask & ~np.isnan(actual)
                    if not train.any():
                        continue
                    mean_train = float(actual[train].mean())
                    sd_train = float(actual[train].std(ddof=0))
                    if sd_train == 0:
                        continue
                    predicted = mean_train + loading * sd_train * score_col
                    valid = row_mask & ~np.isnan(actual) & ~np.isnan(score_col)
                    if not valid.any():
                        continue
                    sse += float(np.sum((actual[valid] - predicted[valid]) ** 2))
                    sso += float(np.sum((actual[valid] - mean_train) ** 2))
            q2 = 1.0 - sse / sso if sso > 0 else np.nan
            rows.append({"lv": target, "q_squared": q2})
        return pd.DataFrame(rows).set_index("lv")
