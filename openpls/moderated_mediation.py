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

"""Index of moderated mediation for PLS-SEM.

A simple mediation ``X -> M -> Y`` is *moderated* when a moderator ``W``
changes the strength of one of its two legs:

- **first stage** (``stage="first"``): ``W`` moderates ``X -> M``. The
  interaction ``X × W -> M`` enters the model and the indirect effect at a
  moderator level ``w`` is ``(a1 + a3·w) · b``.
- **second stage** (``stage="second"``): ``W`` moderates ``M -> Y``. The
  interaction ``M × W -> Y`` enters the model and the indirect effect at
  ``w`` is ``a · (b1 + b3·w)``.

The indirect effect is therefore a linear function of ``w``. Its slope is
the *index of moderated mediation* (Hayes 2015): ``a3 · b`` for first-stage
and ``a · b3`` for second-stage moderation. A bootstrap confidence interval
for the index that excludes zero is the formal test that the indirect effect
depends on the moderator; the conditional indirect effects at chosen
moderator levels describe how.

The interaction term is built with the two-stage approach
(:class:`~openpls.moderation.Moderation`). Moderator levels are in standard
deviations of ``W`` (``-1``, ``0``, ``+1`` = one SD below the mean, the
mean, one SD above), and ``a3`` / ``b3`` are per unit of the product of the
standardized predictor and moderator scores, the scale on which the
conditional effects are linear in ``w``. Bootstrap resamples re-run both
stages, so the uncertainty of the stage-1 scores the interaction term is
built from is part of the interval.

References
----------
- Hayes, A. F. (2015). An index and test of linear moderated mediation.
  Multivariate Behavioral Research, 50(1), 1-22.
- Henseler, J., & Chin, W. W. (2010). A comparison of approaches for the
  analysis of interaction effects between latent variables using partial
  least squares path modeling. Structural Equation Modeling, 17(1).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from openpls.config import Config
from openpls.moderation import Moderation
from openpls.scheme import Scheme

_STAGES = ("first", "second")
DEFAULT_LEVELS = (-1.0, 0.0, 1.0)


def _z(s: pd.Series) -> pd.Series:
    # Sample SD (n - 1), so moderator levels are in the usual SD units.
    return (s - s.mean()) / s.std(ddof=1)


def _extract(moderation: Moderation, roles: dict[str, str], stage: str) -> dict[str, float]:
    """Path coefficients the index is built from: ``a``, ``b`` and the
    interaction coefficient, read from the stage-2 fit.

    Like every LV, the interaction LV is standardized in the stage-2 fit, so
    its path coefficient is per SD of the *product*. The conditional effect
    ``a1 + a3·w`` needs ``a3`` per unit of the raw product of the
    standardized stage-1 scores (so that ``w`` is in SD units of the
    moderator), in SD units of the target: the standardized coefficient times
    the slope of the interaction LV score on that raw product.
    """
    refit = moderation.refit()
    pc = refit.path_coefficients()
    x, m, y, w = roles["predictor"], roles["mediator"], roles["outcome"], roles["moderator"]
    interaction = moderation.interaction_name
    moderated_source, moderated_target = (x, m) if stage == "first" else (m, y)
    ix_std = float(pc.loc[moderated_target, interaction])

    refit_scores = refit.scores()
    base_scores = moderation.base().scores()
    product = (_z(base_scores[moderated_source]) * _z(base_scores[w])).loc[refit_scores.index]
    ix_score = refit_scores[interaction]
    scale = float(
        ((ix_score - ix_score.mean()) * (product - product.mean())).mean()
        / product.var(ddof=0)
        / refit_scores[moderated_target].std(ddof=1)
    )
    return {
        "a": float(pc.loc[m, x]),
        "b": float(pc.loc[y, m]),
        "interaction": ix_std * scale,
        "interaction (standardized)": ix_std,
    }


def _index_of(coef: dict[str, float], stage: str) -> float:
    if stage == "first":
        return coef["interaction"] * coef["b"]
    return coef["a"] * coef["interaction"]


def _conditional_of(coef: dict[str, float], stage: str, level: float) -> float:
    if stage == "first":
        return (coef["a"] + coef["interaction"] * level) * coef["b"]
    return coef["a"] * (coef["b"] + coef["interaction"] * level)


def _bootstrap_chunk(
    data: pd.DataFrame,
    config: Config,
    moderation_kwargs: dict,
    roles: dict[str, str],
    stage: str,
    levels: list[float],
    row_draws: list[np.ndarray],
) -> list[list[float] | None]:
    """Fit one batch of resamples. Returns ``[index, *conditional]`` per
    resample, or ``None`` where the fit failed or was not finite."""
    out: list[list[float] | None] = []
    for rows in row_draws:
        sample = data.iloc[rows].reset_index(drop=True)
        try:
            coef = _extract(Moderation(sample, config, **moderation_kwargs), roles, stage)
        except Exception:
            out.append(None)
            continue
        values = [_index_of(coef, stage)] + [_conditional_of(coef, stage, w) for w in levels]
        out.append(values if np.all(np.isfinite(values)) else None)
    return out


def _summarise(original: float, samples: np.ndarray, alpha: float) -> dict:
    std = float(np.std(samples, ddof=1))
    return {
        "original": original,
        "mean": float(np.mean(samples)),
        "std.error": std,
        "perc.lower": float(np.quantile(samples, alpha / 2.0)),
        "perc.upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "t stat.": original / std if std > 0 else np.nan,
    }


class ModeratedMediation:
    """Index of moderated mediation for ``predictor -> mediator -> outcome``.

    Parameters
    ----------
    data : pd.DataFrame
        The original dataset.
    config : Config
        The base-model config, without the interaction term. It must contain
        the paths ``predictor -> mediator`` and ``mediator -> outcome``, plus
        the main effect of the moderator on the moderated path's target
        (``moderator -> mediator`` for ``stage="first"``,
        ``moderator -> outcome`` for ``stage="second"``). Without that main
        effect the interaction coefficient absorbs it and the index is biased.
    predictor, mediator, outcome, moderator : str
        Latent variable names (all distinct).
    stage : {"first", "second"}
        Which leg of the mediation the moderator acts on.
    interaction_name : str, optional
        Name of the interaction LV. Defaults to ``"{X}_x_{W}"`` (first stage)
        or ``"{M}_x_{W}"`` (second stage).
    scheme, iterations, tolerance, missing_strategy
        Passed through to every :class:`~openpls.plspm.Plspm` fit.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        predictor: str,
        mediator: str,
        outcome: str,
        moderator: str,
        stage: str = "first",
        interaction_name: str | None = None,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 100,
        tolerance: float = 1e-6,
        missing_strategy: str = "casewise",
    ):
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
        path = config.path()
        roles = {
            "predictor": predictor,
            "mediator": mediator,
            "outcome": outcome,
            "moderator": moderator,
        }
        for role, name in roles.items():
            if name not in path.index:
                raise ValueError(f"{role} {name!r} is not a latent variable in the model")
        if len(set(roles.values())) != len(roles):
            raise ValueError("predictor, mediator, outcome and moderator must be different LVs")
        for source, target in ((predictor, mediator), (mediator, outcome)):
            if path.loc[target, source] != 1:
                raise ValueError(
                    f"the mediation needs the path {source} -> {target} in the model"
                )
        moderated_target = mediator if stage == "first" else outcome
        if path.loc[moderated_target, moderator] != 1:
            raise ValueError(
                f"add the main effect {moderator} -> {moderated_target} before "
                "moderating the path; without it the index is biased"
            )

        self.__data = data
        self.__config = config
        self.__roles = roles
        self.__stage = stage
        self.__moderation_kwargs = {
            "predictor": predictor if stage == "first" else mediator,
            "moderator": moderator,
            "target": moderated_target,
            "interaction_name": interaction_name,
            "scheme": scheme,
            "iterations": iterations,
            "tolerance": tolerance,
            "missing_strategy": missing_strategy,
        }

        self.__moderation = Moderation(data, config, **self.__moderation_kwargs)
        self.__coefficients = _extract(self.__moderation, roles, stage)

    @property
    def stage(self) -> str:
        return self.__stage

    @property
    def interaction_name(self) -> str:
        return self.__moderation.interaction_name

    def moderation(self) -> Moderation:
        """The underlying two-stage moderation fit."""
        return self.__moderation

    def coefficients(self) -> pd.Series:
        """Path coefficients behind the index.

        ``a`` (predictor -> mediator), ``b`` (mediator -> outcome) and
        ``interaction`` (``a3`` for first-stage, ``b3`` for second-stage
        moderation), all from the fit that includes the interaction term.
        ``interaction`` is per unit of the product of the standardized
        predictor and moderator scores, the scale the index and the
        conditional effects use; ``interaction (standardized)`` is the path
        coefficient of the standardized interaction LV as the model reports it.
        """
        return pd.Series(self.__coefficients, name="estimate")

    def index(self) -> float:
        """Index of moderated mediation (point estimate)."""
        return _index_of(self.__coefficients, self.__stage)

    def conditional_indirect_effects(
        self, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> pd.DataFrame:
        """Indirect effect at each moderator level (in SD units of ``W``)."""
        rows = [
            {"level": float(w), "estimate": _conditional_of(self.__coefficients, self.__stage, w)}
            for w in levels
        ]
        return pd.DataFrame(rows).set_index("level")

    def bootstrap(
        self,
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: int | None = None,
        levels: Sequence[float] = DEFAULT_LEVELS,
        n_jobs: int = 1,
    ) -> ModeratedMediationBootstrap:
        """Percentile bootstrap for the index and the conditional effects.

        Each resample draws rows with replacement and re-runs both stages of
        the moderation. Resamples whose fit fails are dropped and counted in
        :attr:`ModeratedMediationBootstrap.n_failed`.

        ``n_jobs > 1`` spreads the resamples over that many worker processes.
        All resamples are drawn up front from ``seed``, so the result does not
        depend on ``n_jobs``.
        """
        if n_boot < 2:
            raise ValueError(f"n_boot must be at least 2, got {n_boot}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
        levels = [float(w) for w in levels]
        rng = np.random.default_rng(seed)
        n = self.__data.shape[0]
        draws = [rng.integers(0, n, size=n) for _ in range(n_boot)]

        args = (self.__data, self.__config, self.__moderation_kwargs, self.__roles, self.__stage, levels)
        if n_jobs == 1:
            results = _bootstrap_chunk(*args, draws)
        else:
            chunks = [c for c in np.array_split(np.arange(n_boot), n_jobs) if len(c)]
            with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
                futures = [
                    pool.submit(_bootstrap_chunk, *args, [draws[i] for i in chunk])
                    for chunk in chunks
                ]
                results = [r for f in futures for r in f.result()]

        valid = np.asarray([r for r in results if r is not None])
        if len(valid) < 2:
            raise RuntimeError(
                f"only {len(valid)} of {n_boot} bootstrap resamples could be estimated"
            )

        index_summary = pd.Series(
            _summarise(self.index(), valid[:, 0], alpha),
            name="index of moderated mediation",
        )
        cond_rows = []
        for j, w in enumerate(levels, start=1):
            original = _conditional_of(self.__coefficients, self.__stage, w)
            cond_rows.append({"level": w, **_summarise(original, valid[:, j], alpha)})
        cond_summary = pd.DataFrame(cond_rows).set_index("level")
        return ModeratedMediationBootstrap(
            index_summary,
            cond_summary,
            n_valid=len(valid),
            n_failed=n_boot - len(valid),
            alpha=alpha,
        )


class ModeratedMediationBootstrap:
    """Bootstrap results of :meth:`ModeratedMediation.bootstrap`."""

    def __init__(
        self,
        index: pd.Series,
        conditional: pd.DataFrame,
        n_valid: int,
        n_failed: int,
        alpha: float,
    ):
        self.__index = index
        self.__conditional = conditional
        self.n_valid = n_valid
        self.n_failed = n_failed
        self.alpha = alpha

    def index(self) -> pd.Series:
        """Index of moderated mediation: ``original``, ``mean``,
        ``std.error``, ``perc.lower``, ``perc.upper`` (at level
        ``1 - alpha``) and ``t stat.``."""
        return self.__index

    def conditional_indirect_effects(self) -> pd.DataFrame:
        """Conditional indirect effects per moderator level, same columns."""
        return self.__conditional

    def significant(self) -> bool:
        """``True`` if the percentile CI of the index excludes zero, i.e. the
        indirect effect depends on the moderator."""
        lo = self.__index["perc.lower"]
        hi = self.__index["perc.upper"]
        return bool(lo > 0 or hi < 0)
