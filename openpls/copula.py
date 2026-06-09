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

import numpy as np
import pandas as pd
from scipy import stats

import openpls.config as c


def _copula_term(x: np.ndarray) -> np.ndarray:
    """Park & Gupta (2012) Gaussian copula augmentation term.

    Constructs ``P = Phi^{-1}(F_n(x))``, where ``F_n`` is the empirical
    CDF rescaled to ``(0, 1)`` via ``rank / (n + 1)`` to keep the
    transformation finite at the boundaries.
    """
    n = x.shape[0]
    if n < 3:
        raise ValueError("Gaussian copula requires at least 3 observations")
    ranks = stats.rankdata(x, method="average")
    u = ranks / (n + 1.0)
    return stats.norm.ppf(u)


def _ols(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS coefficient vector for ``y ~ [1, x]``. Returns coefficients
    excluding the intercept."""
    n = y.shape[0]
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta[1:]


class GaussianCopula:
    """Gaussian copula approach for endogeneity in PLS-SEM
    (Park & Gupta 2012; Hult, Hair, Proksch, Sarstedt, Pinkwart &
    Ringle 2018).

    For a chosen endogenous latent variable ``Y`` with predecessor LVs
    ``X_1, ..., X_p``, each suspected endogenous predictor ``X_k`` is
    augmented in the structural regression with a copula term
    ``P_k = Phi^{-1}(F_n(X_k))``. A statistically significant copula
    coefficient ``gamma_k`` indicates that ``X_k`` is correlated with
    the omitted-variable error and therefore likely endogenous.

    The augmented model::

        Y = beta_0 + sum_j beta_j X_j + sum_{k in suspected} gamma_k P_k + e

    is estimated by OLS on the latent-variable scores. Inference on the
    copula coefficients is obtained by a non-parametric bootstrap: rows
    are resampled with replacement, the augmented regression is refit,
    and the standard error is the empirical standard deviation of the
    bootstrap distribution. The two-sided p-value uses the normal-
    approximation ``t = gamma / SE`` and ``p = 2 * (1 - Phi(|t|))``, the
    same convention used by :class:`.long_bootstrap.LongBootstrap`.

    The procedure assumes the suspected predictor is *non-normal*: the
    copula term degenerates under a Gaussian regressor because
    ``Phi^{-1}(F_n(X))`` is then approximately identical to ``X`` itself.
    Each tested predictor is screened with the Cramér-von Mises test
    against the empirical-mean / empirical-sd normal distribution and
    the p-value is reported alongside the coefficient. A small CvM
    p-value (e.g. ``< 0.05``) means non-normality is supported and the
    copula approach is admissible (Hult et al. 2018).
    """

    def __init__(
        self,
        config: c.Config,
        scores: pd.DataFrame,
        endogenous: str,
        suspected: list[str] | None = None,
        n_boot: int = 500,
        seed: int | None = 42,
    ):
        path = config.path()
        if endogenous not in path.index:
            raise ValueError(
                f"endogenous LV {endogenous!r} not found in the structural model"
            )
        predictors = [lv for lv in path.columns if path.loc[endogenous, lv] == 1]
        if not predictors:
            raise ValueError(
                f"LV {endogenous!r} has no predecessors; nothing to test"
            )
        if suspected is None:
            suspected = list(predictors)
        unknown = [lv for lv in suspected if lv not in predictors]
        if unknown:
            raise ValueError(
                f"suspected LVs {unknown!r} are not predecessors of {endogenous!r}"
            )
        if n_boot < 50:
            raise ValueError("n_boot must be at least 50")

        y = scores[endogenous].to_numpy(dtype=float)
        x_full = scores[predictors].to_numpy(dtype=float)
        n = y.shape[0]

        copula_terms = {lv: _copula_term(scores[lv].to_numpy(dtype=float)) for lv in suspected}

        normality = {}
        for lv in suspected:
            arr = scores[lv].to_numpy(dtype=float)
            cvm = stats.cramervonmises(
                arr,
                "norm",
                args=(float(arr.mean()), float(arr.std(ddof=1))),
            )
            normality[lv] = float(cvm.pvalue)

        augment = np.column_stack([copula_terms[lv] for lv in suspected])
        design = np.column_stack([x_full, augment])
        coef_full = _ols(y, design)
        coef_paths = pd.Series(
            coef_full[: len(predictors)], index=predictors, name="estimate"
        )
        coef_copulas = pd.Series(
            coef_full[len(predictors) :], index=suspected, name="estimate"
        )

        rng = np.random.default_rng(seed)
        boot_copulas = np.empty((n_boot, len(suspected)))
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                coef_b = _ols(y[idx], design[idx])
            except np.linalg.LinAlgError:
                boot_copulas[b, :] = np.nan
                continue
            boot_copulas[b, :] = coef_b[len(predictors) :]

        valid_mask = ~np.isnan(boot_copulas).any(axis=1)
        boot_valid = boot_copulas[valid_mask]
        if boot_valid.shape[0] < 50:
            raise RuntimeError(
                f"only {boot_valid.shape[0]} bootstrap fits succeeded; "
                "the augmented regression may be near-singular"
            )

        se = boot_valid.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(se > 0, coef_copulas.values / se, np.nan)
        p_vals = np.where(
            np.isnan(t_vals), np.nan, 2.0 * (1.0 - stats.norm.cdf(np.abs(t_vals)))
        )

        coef_table = pd.DataFrame(
            {
                "predictor": suspected,
                "gamma": coef_copulas.values,
                "boot_se": se,
                "t": t_vals,
                "p_value": p_vals,
                "cvm_p_nonnormal": [normality[lv] for lv in suspected],
            }
        )

        self.__endogenous = endogenous
        self.__predictors = predictors
        self.__suspected = list(suspected)
        self.__n_boot = int(boot_valid.shape[0])
        self.__alpha = 0.05
        self.__augmented_paths = coef_paths
        self.__coefficients = coef_table

    def endogenous(self) -> str:
        """The endogenous LV under test."""
        return self.__endogenous

    def predictors(self) -> list[str]:
        """All structural predecessors of the endogenous LV."""
        return list(self.__predictors)

    def suspected(self) -> list[str]:
        """Predictors that received a Gaussian-copula augmentation term."""
        return list(self.__suspected)

    def coefficients(self) -> pd.DataFrame:
        """Per-predictor copula diagnostics.

        Columns: ``predictor``, ``gamma`` (the copula coefficient in the
        augmented regression), ``boot_se`` (bootstrap standard error),
        ``t`` and ``p_value`` (two-sided normal-approximation), and
        ``cvm_p_nonnormal`` (Cramér-von Mises p-value of the predictor
        against a normal with its sample mean / sample sd; small means
        non-normality is supported and the copula approach is
        admissible).
        """
        return self.__coefficients

    def augmented_paths(self) -> pd.Series:
        """Path coefficients of the endogenous LV's structural equation
        *with* the copula terms in the model. These are the endogeneity-
        corrected estimates that should be compared with the original
        :meth:`Plspm.path_coefficients` to gauge the magnitude of the
        endogeneity bias.
        """
        return self.__augmented_paths

    def summary(self) -> pd.DataFrame:
        """Per-predictor decision summary.

        Adds a ``decision`` column to :meth:`coefficients`:

        * ``"endogeneity detected"`` if ``p_value <= alpha`` and the
          Cramér-von Mises non-normality test rejects normality at
          ``alpha``;
        * ``"copula not admissible (normal)"`` if the Cramér-von Mises
          test fails to reject normality at ``alpha`` — the test
          cannot distinguish endogeneity from a Gaussian regressor;
        * ``"no endogeneity detected"`` otherwise.
        """
        alpha = self.__alpha
        out = self.__coefficients.copy()
        decisions: list[str] = []
        for _, row in out.iterrows():
            if row["cvm_p_nonnormal"] > alpha:
                decisions.append("copula not admissible (normal)")
            elif np.isnan(row["p_value"]):
                decisions.append("inconclusive")
            elif row["p_value"] <= alpha:
                decisions.append("endogeneity detected")
            else:
                decisions.append("no endogeneity detected")
        out["decision"] = decisions
        return out

    def n_boot(self) -> int:
        """Number of successful bootstrap iterations used for SE / p."""
        return self.__n_boot
