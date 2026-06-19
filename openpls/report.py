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

"""Publication-ready summary report for a fitted PLS-SEM model.

Bundles the engine's individual diagnostics — reliability (Cronbach
alpha, rho_A, rho_C, AVE), discriminant validity (HTMT and Fornell-
Larcker), structural paths with f² effect sizes and R² / adjusted R² /
BIC per endogenous LV, fit indices (SRMR, d_ULS, GoF), and collinearity
(outer VIF, inner VIF) — into one place so the standard PLS-SEM
research report (Hair, Hult, Ringle and Sarstedt 2022, *A Primer on
PLS-SEM*, 3rd ed.) can be exported with a single call.

This module is pure orchestration; every value comes from an existing
``Plspm`` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openpls.plspm import Plspm


class Report:
    """Publication-ready summary of a fitted PLS-SEM model.

    Constructed via :meth:`~openpls.plspm.Plspm.report`.

    Methods return prepared DataFrames covering the standard reporting
    panels. :meth:`to_dict` bundles everything for export.
    """

    def __init__(self, plspm: Plspm, include_rho_a: bool = True, include_htmt2: bool = True):
        self._plspm = plspm
        self._include_rho_a = include_rho_a
        self._include_htmt2 = include_htmt2

    def reliability(self) -> pd.DataFrame:
        """Reliability and convergent validity per LV.

        Columns: ``mode``, ``mvs``, ``cronbach_alpha``, ``rho_a``,
        ``rho_c``, ``ave``. Mode B (formative) and single-indicator LVs
        receive ``NaN`` for the metrics that are undefined for them.
        """
        rel = self._plspm.reliability()
        inner = self._plspm.inner_summary()
        out = pd.DataFrame({
            "mode": rel["mode"],
            "mvs": rel["mvs"].astype("Int64"),
            "cronbach_alpha": rel["cronbach_alpha"],
            "rho_c": rel["dillon_goldstein_rho"],
            "ave": inner["ave"],
        })
        if self._include_rho_a:
            try:
                rho_a = self._plspm.plsc().rho_a()
                out.insert(3, "rho_a", rho_a.reindex(out.index))
            except Exception:
                out.insert(3, "rho_a", np.nan)
        return out

    def discriminant_validity(self) -> dict[str, pd.DataFrame]:
        """HTMT / HTMT2 and Fornell-Larcker matrices.

        Keys: ``htmt`` (matrix), ``htmt_pairs`` (long form),
        ``fornell_larcker`` (matrix), ``fornell_larcker_summary``
        (per-LV passes verdict), and (when ``include_htmt2``) ``htmt2``
        and ``htmt2_pairs``.
        """
        out: dict[str, pd.DataFrame] = {}
        htmt = self._plspm.htmt()
        out["htmt"] = htmt.matrix()
        out["htmt_pairs"] = htmt.pairs()
        if self._include_htmt2:
            htmt2 = self._plspm.htmt2()
            out["htmt2"] = htmt2.matrix()
            out["htmt2_pairs"] = htmt2.pairs()
        fl = self._plspm.fornell_larcker()
        out["fornell_larcker"] = fl.matrix()
        out["fornell_larcker_summary"] = fl.summary()
        return out

    def paths(self) -> pd.DataFrame:
        """Structural-model paths with f² and effect-size labels.

        Indexed by ``"predictor -> endogenous"``. Columns: ``from``,
        ``to``, ``estimate``, ``std_error``, ``t``, ``p_value``,
        ``f_squared``, ``effect_size``.
        """
        inner = self._plspm.inner_model().copy()
        inner = inner.rename(columns={"std error": "std_error", "p>|t|": "p_value"})
        f2 = self._plspm.f_squared().table().reindex(inner.index)
        inner["f_squared"] = f2["f_squared"]
        inner["effect_size"] = f2["effect_size"]
        return inner.loc[:, [
            "from", "to", "estimate", "std_error", "t", "p_value",
            "f_squared", "effect_size",
        ]]

    def construct_summary(self) -> pd.DataFrame:
        """Per-LV structural summary.

        Columns: ``type`` (Exogenous / Endogenous), ``mvs``,
        ``r_squared``, ``r_squared_adj``, ``bic``,
        ``block_communality``, ``mean_redundancy``.
        """
        inner = self._plspm.inner_summary().copy()
        rel = self._plspm.reliability()
        inner["mvs"] = rel["mvs"].astype("Int64")
        return inner.loc[:, [
            "type", "mvs", "r_squared", "r_squared_adj", "bic",
            "block_communality", "mean_redundancy",
        ]]

    def fit_indices(self) -> pd.Series:
        """Model fit indices.

        Series with ``srmr``, ``d_uls``, ``goodness_of_fit`` (NaN if all
        constructs are single-item).
        """
        mf = self._plspm.model_fit()
        try:
            gof = float(self._plspm.goodness_of_fit())
        except ValueError:
            gof = float("nan")
        return pd.Series({
            "srmr": float(mf.srmr()),
            "d_uls": float(mf.d_uls()),
            "goodness_of_fit": gof,
        }, name="fit_indices")

    def collinearity(self) -> dict[str, object]:
        """Outer and inner VIF.

        Keys: ``items`` (per-indicator outer VIF; may be ``None`` if no
        block has at least two indicators), ``inner`` (dict of per-
        endogenous-LV VIF tables).
        """
        vif = self._plspm.vif()
        return {"items": vif.items(), "inner": vif.inner()}

    def to_dict(self) -> dict[str, object]:
        """Bundle every section into a single dictionary, ready for
        export (e.g. to JSON via :func:`pandas.DataFrame.to_dict` on
        each value)."""
        return {
            "reliability": self.reliability(),
            "discriminant_validity": self.discriminant_validity(),
            "paths": self.paths(),
            "construct_summary": self.construct_summary(),
            "fit_indices": self.fit_indices(),
            "collinearity": self.collinearity(),
        }

    def __repr__(self) -> str:
        rel = self.reliability()
        fit = self.fit_indices()
        n_paths = len(self.paths())
        return (
            f"<openpls.Report — {len(rel)} LVs, {n_paths} structural paths, "
            f"SRMR={fit['srmr']:.4f}, d_ULS={fit['d_uls']:.4f}>"
        )
