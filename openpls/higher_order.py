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

"""Disjoint two-stage higher-order construct (HOC) workflow for PLS-SEM.

Implements the disjoint two-stage approach recommended by
Sarstedt, Hair, Cheah, Becker & Ringle (2019) and Hair, Hult, Ringle &
Sarstedt (2022, A primer on PLS-SEM, 3rd ed., Chapter 8):

1. **Stage 1**: Fit the lower-order model in which the constituent
   first-order constructs appear as siblings (no second-order LV).
2. **Stage 2**: Use the standardized first-order LV scores from stage 1
   as the **indicators** of the second-order construct, and refit a
   model where the HOC replaces its first-order constructs in the
   structural part of the model.

This is statistically cleaner than the legacy ``Config.add_higher_order``
flow (repeated-indicators / embedded two-stage), because in the disjoint
version the first-order constructs are not simultaneously their own
measurement and the HOC's predictors. The four canonical HOC types are
covered via the ``mode`` argument and the modes already set on the
first-order LVs:

* **Reflective-Reflective (Type I)**: first-order Mode A, HOC Mode A.
* **Reflective-Formative (Type II)**: first-order Mode A, HOC Mode B.
* **Formative-Reflective (Type III)**: first-order Mode B, HOC Mode A.
* **Formative-Formative (Type IV)**: first-order Mode B, HOC Mode B.

References
----------
- Sarstedt, M., Hair, J. F., Cheah, J.-H., Becker, J.-M., & Ringle, C. M.
  (2019). How to specify, estimate, and validate higher-order constructs
  in PLS-SEM. Australasian Marketing Journal, 27(3), 197-211.
- Hair, J. F., Hult, G. T. M., Ringle, C. M., & Sarstedt, M. (2022).
  A primer on partial least squares structural equation modeling
  (PLS-SEM), 3rd ed., Chapter 8.
"""

from __future__ import annotations

import pandas as pd

from openpls.config import MV, Config, Structure
from openpls.mode import Mode
from openpls.scheme import Scheme


def _build_stage2_config(
    base_config: Config,
    hoc_name: str,
    first_order: list[str],
    hoc_mode: Mode,
    structure: Structure,
    score_columns: dict[str, str],
) -> Config:
    """Build the stage-2 Config: HOC has the first-order LV scores as
    indicators; every other LV reuses its original indicators."""
    stage2_path = structure.path()
    if hoc_name not in stage2_path.index:
        raise ValueError(
            f"HOC {hoc_name!r} does not appear in the stage-2 structure"
        )
    for lo in first_order:
        if lo in stage2_path.index:
            raise ValueError(
                f"first-order LV {lo!r} must NOT appear in the stage-2 "
                f"structure (it has been rolled into {hoc_name!r})"
            )

    stage2 = Config(stage2_path, scaled=base_config.scaled())
    for lv in stage2_path.index:
        if lv == hoc_name:
            stage2.add_lv(
                hoc_name,
                hoc_mode,
                *[MV(score_columns[lo]) for lo in first_order],
            )
            continue
        if lv not in base_config.path().index:
            raise ValueError(
                f"stage-2 LV {lv!r} is not in the base config; HigherOrder "
                "cannot synthesize indicators for new constructs"
            )
        stage2.add_lv(
            lv,
            base_config.mode(lv),
            *[MV(name) for name in base_config.mvs(lv)],
        )
    return stage2


class HigherOrder:
    """Disjoint two-stage higher-order construct (Hair et al. 2022).

    The base ``Plspm`` fit serves as **stage 1**. The constructor extracts
    its scores for the ``first_order`` LVs, appends them as columns to
    the data, and fits a **stage-2** ``Plspm`` in which:

    * the HOC has the first-order LV scores as its manifest variables
      (mode = ``mode``), and
    * every other LV from the base model that still appears in the
      stage-2 ``structure`` keeps its original indicators.

    The four canonical HOC types are obtained by combining the first-
    order LV modes (set on ``base_config``) with the HOC ``mode``:

    * Type I  (R-R): first-order A, ``mode=Mode.A``;
    * Type II (R-F): first-order A, ``mode=Mode.B``;
    * Type III(F-R): first-order B, ``mode=Mode.A``;
    * Type IV (F-F): first-order B, ``mode=Mode.B``.

    Parameters
    ----------
    base
        A fitted :class:`.plspm.Plspm` whose LV scores will become the
        HOC's indicators.
    name
        Name of the new second-order construct. Must not collide with an
        existing LV or with any indicator column in the data.
    first_order
        Names of the first-order LVs to roll up. All must exist in the
        base model and be exposed in ``base.scores()``.
    mode
        Mode of the HOC's measurement w.r.t. its first-order indicators.
    structure
        :class:`.config.Structure` of the stage-2 path model. Must
        contain ``name`` and may contain any other LV that was in the
        base model except the ones listed in ``first_order``.
    scheme, iterations, tolerance, missing_strategy
        Passed through to the stage-2 :class:`.plspm.Plspm` fit.
    """

    def __init__(
        self,
        base,
        name: str,
        first_order: list[str],
        mode: Mode,
        structure: Structure,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 100,
        tolerance: float = 1e-6,
        missing_strategy: str = "casewise",
    ):
        from openpls.plspm import Plspm  # avoid circular import

        if not isinstance(base, Plspm):
            raise TypeError("base must be a fitted Plspm instance")
        if not isinstance(structure, Structure):
            raise TypeError("structure must be a Structure instance")
        if len(first_order) < 2:
            raise ValueError("a HOC must roll up at least two first-order LVs")
        if len(set(first_order)) != len(first_order):
            raise ValueError("first_order contains duplicate LV names")

        base_config = base.config()
        base_path = base_config.path()
        for lo in first_order:
            if lo not in base_path.index:
                raise ValueError(
                    f"first-order LV {lo!r} is not in the base model"
                )

        scores = base.scores()
        missing_scores = [lo for lo in first_order if lo not in scores.columns]
        if missing_scores:
            raise RuntimeError(
                f"base fit did not produce scores for {missing_scores!r}"
            )

        if name in base_path.index:
            raise ValueError(
                f"HOC name {name!r} clashes with an existing LV in the base model"
            )

        data = base.data()
        score_columns: dict[str, str] = {}
        for lo in first_order:
            col = f"{name}__{lo}"
            if col in data.columns:
                raise ValueError(
                    f"derived indicator column {col!r} already exists in the "
                    "original data; choose a different HOC name"
                )
            score_columns[lo] = col

        data2 = data.copy()
        for lo, col in score_columns.items():
            data2[col] = scores[lo].reindex(data2.index)

        stage2_config = _build_stage2_config(
            base_config=base_config,
            hoc_name=name,
            first_order=first_order,
            hoc_mode=mode,
            structure=structure,
            score_columns=score_columns,
        )

        refit = Plspm(
            data2,
            stage2_config,
            scheme,
            iterations=iterations,
            tolerance=tolerance,
            missing_strategy=missing_strategy,
        )

        self.__base = base
        self.__refit = refit
        self.__name = name
        self.__first_order = list(first_order)
        self.__hoc_mode = mode
        self.__score_columns = score_columns

    def name(self) -> str:
        """Name of the higher-order construct."""
        return self.__name

    def first_order(self) -> list[str]:
        """The first-order LVs rolled into the HOC."""
        return list(self.__first_order)

    def hoc_mode(self) -> Mode:
        """Measurement mode of the HOC w.r.t. its first-order indicators."""
        return self.__hoc_mode

    def base(self):
        """The stage-1 ``Plspm`` fit (first-order siblings)."""
        return self.__base

    def refit(self):
        """The stage-2 ``Plspm`` fit (HOC in place of its first-order LVs)."""
        return self.__refit

    def indicator_columns(self) -> dict[str, str]:
        """Mapping ``first_order_lv -> indicator column name`` used to
        carry the stage-1 scores into the stage-2 data."""
        return dict(self.__score_columns)

    def loadings(self) -> pd.Series:
        """Outer loadings (Mode A) or outer weights (Mode B) of the HOC
        on its first-order indicators, indexed by first-order LV name."""
        outer = self.__refit.outer_model()
        col = "loading" if self.__hoc_mode == Mode.A else "weight"
        out = pd.Series(index=self.__first_order, dtype=float, name=col)
        for lo, ind in self.__score_columns.items():
            out.loc[lo] = float(outer.loc[ind, col])
        return out

    def path_coefficients(self) -> pd.DataFrame:
        """Stage-2 structural path coefficients (HOC included)."""
        return self.__refit.path_coefficients()

    def r_squared(self) -> pd.Series:
        """Stage-2 R² per endogenous LV (HOC included if endogenous)."""
        summary = self.__refit.inner_summary()
        return summary["r_squared"]

    def summary(self) -> pd.DataFrame:
        """Per-first-order-LV measurement summary of the HOC: loading or
        weight (depending on ``hoc_mode``), and the stage-1 R² of that
        first-order LV (NaN for exogenous first-order LVs)."""
        base_r2 = self.__base.inner_summary()["r_squared"]
        col = "loading" if self.__hoc_mode == Mode.A else "weight"
        rows = []
        for lo in self.__first_order:
            ind = self.__score_columns[lo]
            outer = self.__refit.outer_model().loc[ind]
            rows.append(
                {
                    "first_order": lo,
                    col: float(outer[col]),
                    "stage1_r_squared": float(base_r2.get(lo, float("nan"))),
                }
            )
        return pd.DataFrame(rows)
