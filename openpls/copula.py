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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

import openpls.config as c
from openpls.mode import Mode
from openpls.scheme import Scheme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openpls.plspm import Plspm


ALGORITHMS = ("ols_hult", "augmented_plssem")


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


def _gc_indicator_name(lv: str) -> str:
    return f"__gc_{lv}"


def _gc_lv_name(lv: str) -> str:
    return f"GC_{lv}"


class GaussianCopula:
    """Gaussian copula approach for endogeneity in PLS-SEM
    (Park & Gupta 2012; Hult, Hair, Proksch, Sarstedt, Pinkwart &
    Ringle 2018).

    Two algorithms are supported, selected via the ``algorithm``
    argument:

    * ``"ols_hult"`` (default; Hult et al. 2018 one-step OLS variant):
      For the structural equation of ``endogenous``, each suspected
      predecessor LV ``X_k`` is augmented with a copula term
      ``P_k = Phi^{-1}(F_n(X_k))`` inside an OLS-augmented path
      equation on the *base* latent-variable scores. Inference on the
      copula coefficients uses a non-parametric row bootstrap of the
      augmented OLS. A statistically significant copula coefficient
      ``gamma_k`` indicates that ``X_k`` is correlated with the
      omitted-variable error and therefore likely endogenous. The
      augmented model::

          Y = beta_0 + sum_j beta_j X_j + sum_{k in suspected} gamma_k P_k + e

      is estimated by OLS on the latent-variable scores.

    * ``"augmented_plssem"`` (Park & Gupta 2012 augmented-model
      variant): Instead of running OLS on the scores, each copula term
      ``c_k = Phi^{-1}(F_n(X_k))`` is injected as a single-indicator
      *latent variable* ``GC_k`` with a direct path into
      ``endogenous``, and the entire PLS-SEM is refit on the augmented
      model. All paths (original + GC) are reported side-by-side with
      full PLS-SEM standard errors from a non-parametric row bootstrap
      of the augmented refit. Use this variant when replicating
      published examples that report GC paths with PLS-SEM inference
      (rather than with OLS inference on the score-level regression).

    The two approaches yield similar copula-correction magnitudes on
    typical fixtures but report inference differently: ``ols_hult``
    treats the copula terms as extra OLS regressors on the score
    regression, ``augmented_plssem`` treats them as extra latent
    variables in the structural model. The one-step OLS variant is
    faster; the augmented-PLS-SEM variant is more principled when
    downstream reporting expects PLS-SEM standard errors.

    Both algorithms assume the suspected predictor is *non-normal*:
    the copula term degenerates under a Gaussian regressor because
    ``Phi^{-1}(F_n(X))`` is then approximately identical to ``X``
    itself. Each tested predictor is screened with the Cramér-von
    Mises test against the empirical-mean / empirical-sd normal
    distribution and the p-value is reported alongside the
    coefficient. A small CvM p-value (e.g. ``< 0.05``) means
    non-normality is supported and the copula approach is admissible
    (Hult et al. 2018).
    """

    def __init__(
        self,
        config: c.Config,
        scores: pd.DataFrame,
        endogenous: str,
        suspected: Optional[list[str]] = None,
        n_boot: int = 500,
        seed: Optional[int] = 42,
        *,
        algorithm: Literal["ols_hult", "augmented_plssem"] = "ols_hult",
        manifest_data: Optional[pd.DataFrame] = None,
        scheme: Optional[Scheme] = None,
    ):
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}"
            )
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

        # CvM admissibility screen on the (score-level) suspected predictors.
        # Same convention for both algorithms so summary() semantics stay
        # comparable.
        normality: dict[str, float] = {}
        for lv in suspected:
            arr = scores[lv].to_numpy(dtype=float)
            cvm = stats.cramervonmises(
                arr,
                "norm",
                args=(float(arr.mean()), float(arr.std(ddof=1))),
            )
            normality[lv] = float(cvm.pvalue)

        self.__algorithm = algorithm
        self.__endogenous = endogenous
        self.__predictors = predictors
        self.__suspected = list(suspected)
        self.__alpha = 0.05

        if algorithm == "ols_hult":
            self.__fit_ols_hult(
                scores=scores,
                predictors=predictors,
                suspected=suspected,
                normality=normality,
                n_boot=n_boot,
                seed=seed,
            )
        else:
            if manifest_data is None or scheme is None:
                raise ValueError(
                    "algorithm='augmented_plssem' requires manifest_data and "
                    "scheme; call Plspm.copula(..., algorithm='augmented_plssem')"
                )
            self.__fit_augmented_plssem(
                config=config,
                scores=scores,
                manifest_data=manifest_data,
                scheme=scheme,
                endogenous=endogenous,
                predictors=predictors,
                suspected=suspected,
                normality=normality,
                n_boot=n_boot,
                seed=seed,
            )

    # ------------------------------------------------------------------
    # ols_hult (unchanged one-step OLS variant)
    # ------------------------------------------------------------------

    def __fit_ols_hult(
        self,
        scores: pd.DataFrame,
        predictors: list[str],
        suspected: list[str],
        normality: dict[str, float],
        n_boot: int,
        seed: Optional[int],
    ) -> None:
        endogenous = self.__endogenous
        y = scores[endogenous].to_numpy(dtype=float)
        x_full = scores[predictors].to_numpy(dtype=float)
        n = y.shape[0]

        copula_terms = {
            lv: _copula_term(scores[lv].to_numpy(dtype=float)) for lv in suspected
        }
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

        self.__n_boot = int(boot_valid.shape[0])
        self.__augmented_paths = coef_paths
        self.__coefficients = coef_table
        self.__augmented_full_paths: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # augmented_plssem (Park & Gupta 2012 augmented-model variant)
    # ------------------------------------------------------------------

    def __fit_augmented_plssem(
        self,
        config: c.Config,
        scores: pd.DataFrame,
        manifest_data: pd.DataFrame,
        scheme: Scheme,
        endogenous: str,
        predictors: list[str],
        suspected: list[str],
        normality: dict[str, float],
        n_boot: int,
        seed: Optional[int],
    ) -> None:
        # Import inside method to avoid a top-level circular import
        # (openpls.plspm imports openpls.copula).
        from openpls.plspm import Plspm

        aug_data, aug_config = self.__build_augmented(
            config=config,
            scores=scores,
            manifest_data=manifest_data,
            endogenous=endogenous,
            suspected=suspected,
        )
        aug_fit = Plspm(aug_data, aug_config, scheme)
        aug_paths = aug_fit.path_coefficients()

        # Point estimates on the augmented model.
        gc_gammas = pd.Series(
            {lv: float(aug_paths.loc[endogenous, _gc_lv_name(lv)]) for lv in suspected},
            name="estimate",
        )
        # Corrected structural paths — only for the endogenous LV that
        # was augmented. Other structural equations are unchanged in
        # substance, but we return the whole augmented path matrix via
        # augmented_full_paths().
        augmented_paths = pd.Series(
            {lv: float(aug_paths.loc[endogenous, lv]) for lv in predictors},
            name="estimate",
        )

        # Bootstrap: resample manifest rows, rebuild base scores + GC
        # terms + augmented fit each time.
        rng = np.random.default_rng(seed)
        n = manifest_data.shape[0]
        boot_gammas = np.full((n_boot, len(suspected)), np.nan)
        successes = 0
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = manifest_data.iloc[idx].reset_index(drop=True)
            try:
                sample_aug_data, sample_aug_config = self.__build_augmented(
                    config=config,
                    scores=None,
                    manifest_data=sample,
                    endogenous=endogenous,
                    suspected=suspected,
                    scheme=scheme,
                )
                sample_fit = Plspm(sample_aug_data, sample_aug_config, scheme)
                sample_paths = sample_fit.path_coefficients()
            except Exception:
                continue
            row = np.array(
                [
                    float(sample_paths.loc[endogenous, _gc_lv_name(lv)])
                    for lv in suspected
                ]
            )
            boot_gammas[b, :] = row
            successes += 1

        valid_mask = ~np.isnan(boot_gammas).any(axis=1)
        boot_valid = boot_gammas[valid_mask]
        if boot_valid.shape[0] < 50:
            raise RuntimeError(
                f"only {boot_valid.shape[0]} bootstrap fits succeeded; "
                "the augmented PLS-SEM refit may be near-singular"
            )

        se = boot_valid.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = np.where(se > 0, gc_gammas.values / se, np.nan)
        p_vals = np.where(
            np.isnan(t_vals), np.nan, 2.0 * (1.0 - stats.norm.cdf(np.abs(t_vals)))
        )

        coef_table = pd.DataFrame(
            {
                "predictor": suspected,
                "gamma": gc_gammas.values,
                "boot_se": se,
                "t": t_vals,
                "p_value": p_vals,
                "cvm_p_nonnormal": [normality[lv] for lv in suspected],
            }
        )

        self.__n_boot = int(boot_valid.shape[0])
        self.__augmented_paths = augmented_paths
        self.__coefficients = coef_table
        self.__augmented_full_paths = aug_paths

    def __build_augmented(
        self,
        config: c.Config,
        scores: Optional[pd.DataFrame],
        manifest_data: pd.DataFrame,
        endogenous: str,
        suspected: list[str],
        scheme: Optional[Scheme] = None,
    ) -> tuple[pd.DataFrame, c.Config]:
        """Build the augmented ``(data, config)`` pair.

        If ``scores`` is None, the base PLS-SEM is refit on
        ``manifest_data`` to obtain fresh LV scores (used inside the
        bootstrap resampling loop). Otherwise the supplied scores are
        used directly (used for the point-estimate augmented fit).
        """
        from openpls.plspm import Plspm

        if scores is None:
            # Refit the base model on the resampled manifest data to
            # get resample-specific LV scores.
            base_fit = Plspm(manifest_data, config, scheme or Scheme.CENTROID)
            scores = base_fit.scores()

        # 1) Compute copula terms from base scores, aligned to the
        # (filtered) row index used by the base fit.
        gc_columns: dict[str, np.ndarray] = {}
        for lv in suspected:
            arr = scores[lv].to_numpy(dtype=float)
            gc_columns[_gc_indicator_name(lv)] = _copula_term(arr)

        # 2) Extend the manifest DataFrame with the GC indicator
        # columns. We inject them onto rows that the base fit actually
        # used (scores.index). Rows dropped by config.filter never make
        # it into the augmented fit either — Plspm will filter them
        # again on the augmented config.
        aug_data = manifest_data.loc[scores.index].copy()
        for name, values in gc_columns.items():
            aug_data[name] = values

        # 3) Build augmented structure. Structure() re-topo-sorts, so
        # order-of-add_path doesn't matter as long as no cycles are
        # introduced. GC_<lv> is a fresh LV with no predecessors and
        # one outgoing path into `endogenous`.
        path = config.path()
        struct = c.Structure()
        for tgt in path.index:
            for src in path.columns:
                if path.loc[tgt, src] == 1:
                    struct.add_path([src], [tgt])
        for lv in suspected:
            struct.add_path([_gc_lv_name(lv)], [endogenous])

        aug_config = c.Config(struct.path(), scaled=config.scaled())
        # Original LVs — replay the exact same MV-list + mode. We
        # cannot introspect Config.__mv_scales, but Config.mvs(lv) and
        # Config.mode(lv) are public accessors.
        for lv in path.index:
            mvs = list(config.mvs(lv))
            mv_objs = [c.MV(name) for name in mvs]
            aug_config.add_lv(lv, config.mode(lv), *mv_objs)
        # GC LVs — reflective (Mode.A) single-indicator LVs.
        for lv in suspected:
            aug_config.add_lv(
                _gc_lv_name(lv),
                Mode.A,
                c.MV(_gc_indicator_name(lv)),
            )
        return aug_data, aug_config

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def algorithm(self) -> str:
        """The algorithm variant used to fit this instance
        (``"ols_hult"`` or ``"augmented_plssem"``)."""
        return self.__algorithm

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

        Columns: ``predictor``, ``gamma`` (the copula coefficient — an
        OLS regression coefficient in the ``ols_hult`` variant, a
        PLS-SEM structural path in the ``augmented_plssem`` variant),
        ``boot_se`` (bootstrap standard error), ``t`` and ``p_value``
        (two-sided normal-approximation), and ``cvm_p_nonnormal``
        (Cramér-von Mises p-value of the predictor against a normal
        with its sample mean / sample sd; small means non-normality is
        supported and the copula approach is admissible).
        """
        return self.__coefficients

    def augmented_paths(self) -> pd.Series:
        """Path coefficients of the endogenous LV's structural equation
        *with* the copula terms in the model. These are the endogeneity-
        corrected estimates that should be compared with the original
        :meth:`Plspm.path_coefficients` to gauge the magnitude of the
        endogeneity bias.

        Both algorithms expose the same shape: one entry per
        structural predecessor of ``endogenous``.
        """
        return self.__augmented_paths

    def augmented_full_paths(self) -> Optional[pd.DataFrame]:
        """Full augmented-model path-coefficient matrix, including
        the ``GC_*`` LVs (``augmented_plssem`` only). Returns ``None``
        for the ``ols_hult`` variant."""
        return self.__augmented_full_paths

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
