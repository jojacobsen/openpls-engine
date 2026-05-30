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

"""Two-stage moderation for PLS-SEM.

Adds an interaction term ``predictor × moderator → target`` to a fitted
model using the two-stage approach (Henseler & Chin 2010): fit a base
model, extract standardized LV scores for predictor and moderator,
multiply them, and refit the model with the product as a single-indicator
construct pointing at the target.

This is the modern default in commercial PLS-SEM software (Hair et al.
2017). The base model's path coefficients change in the refit only via
the new interaction LV; standard PLS-SEM reporting compares the base
model's path estimates with the refit including the interaction.

References
----------
- Henseler, J., & Chin, W. W. (2010). A comparison of approaches for the
  analysis of interaction effects between latent variables using partial
  least squares path modeling. Structural Equation Modeling, 17(1).
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2017).
  A primer on partial least squares structural equation modeling
  (PLS-SEM), 2nd ed., Chapter 7.
"""

from __future__ import annotations

import pandas as pd

from plspm.config import MV, Config, Structure
from plspm.mode import Mode
from plspm.scheme import Scheme


def _extend_path(
    base_path: pd.DataFrame,
    interaction_name: str,
    target: str,
) -> pd.DataFrame:
    s = Structure(base_path)
    s.add_path([interaction_name], [target])
    return s.path()


def _extend_config(
    base_config: Config,
    extended_path: pd.DataFrame,
    interaction_name: str,
    indicator_name: str,
) -> Config:
    new_config = Config(extended_path, scaled=base_config.scaled())
    for lv in base_config.path().index:
        mode = base_config.mode(lv)
        mvs = [MV(name) for name in base_config.mvs(lv)]
        new_config.add_lv(lv, mode, *mvs)
    new_config.add_lv(interaction_name, Mode.A, MV(indicator_name))
    return new_config


class Moderation:
    """Two-stage moderation: ``predictor × moderator → target``.

    Stage 1 fits the base model on ``data`` and ``config``. Stage 2 takes
    the standardized LV scores for ``predictor`` and ``moderator`` from
    stage 1, multiplies them into a single product column, and refits with
    the product as the manifest variable of a new single-indicator
    construct that has one path into ``target``.

    Parameters
    ----------
    data : pd.DataFrame
        The original dataset.
    config : Config
        The base-model config. Must include ``predictor``, ``moderator``,
        ``target`` as latent variables. (Predictor and moderator do not
        need to already have a path into target.)
    predictor : str
        Name of the predictor LV.
    moderator : str
        Name of the moderator LV.
    target : str
        Name of the target endogenous LV the interaction will point to.
    interaction_name : str, optional
        Name of the new interaction LV (and its single indicator). Defaults
        to ``"{predictor}_x_{moderator}"``. Must not collide with an
        existing LV or indicator name.
    scheme : Scheme
        Inner weighting scheme for both stages (default centroid).
    iterations, tolerance : int, float
        Passed through to both Plspm fits.
    missing_strategy : str
        Passed through to both Plspm fits.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Config,
        predictor: str,
        moderator: str,
        target: str,
        interaction_name: str | None = None,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 100,
        tolerance: float = 1e-6,
        missing_strategy: str = "casewise",
    ):
        from plspm.plspm import Plspm  # local import to avoid circular dependency

        path = config.path()
        for name, role in (
            (predictor, "predictor"),
            (moderator, "moderator"),
            (target, "target"),
        ):
            if name not in path.index:
                raise ValueError(f"{role} {name!r} is not a latent variable in the model")
        if predictor == moderator:
            raise ValueError("predictor and moderator must be different LVs")
        if target in (predictor, moderator):
            raise ValueError("target cannot be the predictor or moderator")
        if path.loc[target].sum() == 0:
            raise ValueError(f"target {target!r} is exogenous; it has no incoming paths")

        if interaction_name is None:
            interaction_name = f"{predictor}_x_{moderator}"
        if interaction_name in path.index:
            raise ValueError(
                f"interaction_name {interaction_name!r} clashes with an existing LV"
            )

        base = Plspm(
            data,
            config,
            scheme,
            iterations=iterations,
            tolerance=tolerance,
            missing_strategy=missing_strategy,
        )
        scores = base.scores()
        if predictor not in scores.columns or moderator not in scores.columns:
            raise RuntimeError(
                "base fit did not produce scores for predictor or moderator"
            )

        indicator_name = f"{interaction_name}__ind"
        if indicator_name in data.columns:
            raise ValueError(
                f"derived indicator column {indicator_name!r} already exists in data"
            )
        product = (scores[predictor] * scores[moderator]).rename(indicator_name)
        data2 = data.copy().join(product, how="left")
        # rows the base fit dropped (missing) will appear here as NaN in the
        # product column; downstream filter() will handle them per missing_strategy.

        extended_path = _extend_path(path, interaction_name, target)
        extended_config = _extend_config(
            config, extended_path, interaction_name, indicator_name
        )

        refit = Plspm(
            data2,
            extended_config,
            scheme,
            iterations=iterations,
            tolerance=tolerance,
            missing_strategy=missing_strategy,
        )

        self.__base = base
        self.__refit = refit
        self.__predictor = predictor
        self.__moderator = moderator
        self.__target = target
        self.__interaction_name = interaction_name

    @property
    def interaction_name(self) -> str:
        return self.__interaction_name

    def base(self):
        """The stage-1 fit (no interaction term)."""
        return self.__base

    def refit(self):
        """The stage-2 fit including the interaction LV and its path."""
        return self.__refit

    def interaction_effect(self) -> pd.Series:
        """Path coefficient + significance for ``interaction → target``.

        Returns a Series with ``estimate``, ``std error``, ``t``, ``p>|t|``.
        Significance values are only meaningful if you bootstrap the refit;
        the OLS-derived values are convenience reporting.
        """
        inner = self.__refit.inner_model()
        idx = f"{self.__interaction_name} -> {self.__target}"
        if idx not in inner.index:
            raise RuntimeError(
                f"refit did not produce an inner-model row for {idx!r}"
            )
        row = inner.loc[idx]
        return row[["estimate", "std error", "t", "p>|t|"]]
