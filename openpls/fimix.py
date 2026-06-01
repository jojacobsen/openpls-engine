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

"""FIMIX-PLS: Finite Mixture PLS for latent class segmentation.

Given a fitted PLS-SEM model with LV scores, FIMIX detects K subgroups
(latent classes) that share the same measurement model but have distinct
structural-model path coefficients. Estimation is via Expectation-
Maximization on the assumption that each endogenous LV follows a mixture
of normal regressions on its direct predecessors.

The procedure is essentially Hahn et al. (2002), the standard FIMIX
formulation used by commercial PLS-SEM tools.

References
----------
- Hahn, C., Johnson, M. D., Herrmann, A., & Huber, F. (2002). Capturing
  customer heterogeneity using a finite mixture PLS approach. Schmalenbach
  Business Review, 54(3), 243-269.
- Sarstedt, M., Becker, J.-M., Ringle, C. M., & Schwaiger, M. (2011).
  Uncovering and treating unobserved heterogeneity with FIMIX-PLS: Which
  model selection criterion provides an appropriate number of segments?
  Schmalenbach Business Review, 63(1), 34-62.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

_LOG2PI = math.log(2.0 * math.pi)


def _endogenous_structure(path: pd.DataFrame) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for lv in path.index:
        preds = [p for p in path.columns if path.loc[lv, p] == 1]
        if preds:
            out.append((lv, preds))
    return out


def _weighted_ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # X already has intercept column. w is shape (n,) of per-row weights.
    sqw = np.sqrt(np.maximum(w, 0.0))
    Xw = X * sqw[:, None]
    yw = y * sqw
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef


class FIMIX:
    """Finite Mixture PLS for ``n_classes`` latent classes.

    Parameters
    ----------
    plspm : Plspm
        A fitted Plspm instance. FIMIX uses its LV scores and path matrix.
    n_classes : int
        Number of mixture components K (>= 2).
    max_iter : int
        Maximum EM iterations per restart (default 500).
    tolerance : float
        Convergence threshold on the log-likelihood between iterations.
    n_restarts : int
        Number of random EM restarts. The best (highest log-likelihood) is
        kept. Default 5.
    seed : int | None
        RNG seed for restart initialization. ``None`` for non-deterministic.
    """

    def __init__(
        self,
        plspm,
        n_classes: int,
        max_iter: int = 500,
        tolerance: float = 1e-6,
        n_restarts: int = 5,
        seed: int | None = 42,
    ):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if max_iter < 10:
            raise ValueError("max_iter must be >= 10")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")

        self.__K = int(n_classes)
        self.__max_iter = max_iter
        self.__tolerance = tolerance
        self.__n_restarts = n_restarts
        self.__seed = seed

        scores = plspm.scores()
        # take a clean copy of LV scores aligned to the path matrix
        path = plspm.path_coefficients()  # rows=endog, columns=all LVs
        # path_coefficients only covers endogenous rows. Reconstruct the structural
        # path matrix the engine actually used by inferring from path_coefficients
        # (any LV in columns is a candidate predecessor for each row).
        endo_structure: list[tuple[str, list[str]]] = []
        for lv in path.index:
            if lv not in scores.columns:
                continue
            preds = [p for p in path.columns if p in scores.columns and abs(float(path.loc[lv, p])) > 0]
            if preds:
                endo_structure.append((lv, preds))
        if not endo_structure:
            raise ValueError("model has no endogenous LV with predecessors; FIMIX needs structural paths")
        self.__endo_structure = endo_structure
        self.__scores = scores
        self.__results: dict | None = None

    @property
    def n_classes(self) -> int:
        return self.__K

    def __em_once(self, init_seed: int | None) -> dict:
        K = self.__K
        scores = self.__scores
        n = len(scores)
        rng = np.random.default_rng(init_seed)
        # initialize posterior assignments via random class draws, then normalize
        post = rng.dirichlet(alpha=np.ones(K), size=n)  # (n, K)
        log_lik_prev = -np.inf

        # Pre-extract design matrices and outcomes per endo LV
        per_lv = []
        for lv, preds in self.__endo_structure:
            X = np.column_stack([np.ones(n), scores[preds].to_numpy(dtype=float)])
            y = scores[lv].to_numpy(dtype=float)
            per_lv.append((lv, preds, X, y))

        rho = np.full(K, 1.0 / K)
        betas: list[np.ndarray] = []  # K-many list of (p_lv+1)-length coefs per LV
        sigmas: list[np.ndarray] = []  # K-many list of variances per LV

        for it in range(self.__max_iter):
            # ---- M-step ----
            rho = post.mean(axis=0)
            rho = np.maximum(rho, 1e-12)
            rho = rho / rho.sum()

            betas = []
            sigmas = []
            for k in range(K):
                w = post[:, k]
                w_sum = max(w.sum(), 1e-12)
                beta_k: list[np.ndarray] = []
                sig_k: list[float] = []
                for lv, preds, X, y in per_lv:
                    coef = _weighted_ols(X, y, w)
                    resid = y - X @ coef
                    var = float(np.sum(w * resid * resid) / w_sum)
                    var = max(var, 1e-10)
                    beta_k.append(coef)
                    sig_k.append(var)
                betas.append(beta_k)
                sigmas.append(np.array(sig_k))

            # ---- E-step ----
            log_resp = np.zeros((n, K))
            for k in range(K):
                lp = np.full(n, math.log(rho[k]))
                for j, (lv, preds, X, y) in enumerate(per_lv):
                    mu = X @ betas[k][j]
                    var = sigmas[k][j]
                    lp = lp - 0.5 * (_LOG2PI + math.log(var)) - 0.5 * ((y - mu) ** 2) / var
                log_resp[:, k] = lp
            max_lp = log_resp.max(axis=1, keepdims=True)
            log_norm = max_lp.squeeze(axis=1) + np.log(
                np.exp(log_resp - max_lp).sum(axis=1)
            )
            post = np.exp(log_resp - log_norm[:, None])
            log_lik = float(log_norm.sum())

            if abs(log_lik - log_lik_prev) < self.__tolerance:
                break
            log_lik_prev = log_lik

        # final parameter count: per class we have (sum_j p_j coefficients including
        # intercept) + (number of endo LVs) variances; plus (K-1) free mixture probs.
        per_class_params = sum(X.shape[1] for _, _, X, _ in per_lv) + len(per_lv)
        n_params = K * per_class_params + (K - 1)

        return {
            "rho": rho,
            "betas": betas,
            "sigmas": sigmas,
            "post": post,
            "log_lik": log_lik,
            "n_params": n_params,
            "iter": it + 1,
        }

    def __ensure_fit(self):
        if self.__results is not None:
            return
        best: dict | None = None
        for r in range(self.__n_restarts):
            seed = (self.__seed + r) if self.__seed is not None else None
            res = self.__em_once(seed)
            if best is None or res["log_lik"] > best["log_lik"]:
                best = res
        assert best is not None
        self.__results = best

    def log_likelihood(self) -> float:
        self.__ensure_fit()
        return float(self.__results["log_lik"])

    def class_sizes(self) -> pd.Series:
        """Mixture proportions ``rho_k`` (one per class)."""
        self.__ensure_fit()
        return pd.Series(
            self.__results["rho"],
            index=[f"class_{k + 1}" for k in range(self.__K)],
            name="rho",
        )

    def memberships(self) -> pd.DataFrame:
        """Posterior class probabilities ``P(class_k | case_i)``.

        Indexed by the original case index, with one column per class.
        """
        self.__ensure_fit()
        return pd.DataFrame(
            self.__results["post"],
            index=self.__scores.index,
            columns=[f"class_{k + 1}" for k in range(self.__K)],
        )

    def hard_assignments(self) -> pd.Series:
        """Hard class label per case (the argmax of the posterior)."""
        post = self.memberships()
        labels = post.values.argmax(axis=1) + 1
        return pd.Series(labels, index=post.index, name="class").astype(int)

    def class_paths(self) -> pd.DataFrame:
        """Class-specific structural path coefficients.

        Long-format DataFrame with columns ``class``, ``from``, ``to``,
        ``estimate``, ``intercept``. One row per (class, endogenous LV,
        predecessor) plus one ``"(intercept)"`` row per (class, endogenous
        LV) for completeness.
        """
        self.__ensure_fit()
        betas = self.__results["betas"]
        rows: list[dict] = []
        for k in range(self.__K):
            for j, (lv, preds, _X, _y) in enumerate(
                [(lv, preds, None, None) for lv, preds in self.__endo_structure]
            ):
                coef = betas[k][j]
                rows.append(
                    {
                        "class": k + 1,
                        "from": "(intercept)",
                        "to": lv,
                        "estimate": float(coef[0]),
                    }
                )
                for p_idx, pred in enumerate(preds):
                    rows.append(
                        {
                            "class": k + 1,
                            "from": pred,
                            "to": lv,
                            "estimate": float(coef[p_idx + 1]),
                        }
                    )
        return pd.DataFrame(rows)

    def fit_criteria(self) -> pd.Series:
        """Information-theoretic model-selection criteria.

        Lower is better for AIC, BIC, CAIC, AIC3, AIC4, MDL5. Higher is
        better for EN (normalized entropy in [0, 1]; closer to 1 means
        clearer class separation).
        """
        self.__ensure_fit()
        ll = self.log_likelihood()
        n = len(self.__scores)
        K = self.__K
        m = self.__results["n_params"]
        post = self.__results["post"]
        # entropy criterion EN (Ramaswamy et al. 1993):
        #   EN = 1 - sum_i sum_k -p_ik log(p_ik) / (n log K)
        clipped = np.clip(post, 1e-12, 1.0)
        ent = -float(np.sum(clipped * np.log(clipped)))
        en = 1.0 - ent / (n * math.log(K)) if K > 1 else float("nan")
        return pd.Series(
            {
                "log_lik": ll,
                "n_params": m,
                "aic": -2 * ll + 2 * m,
                "aic3": -2 * ll + 3 * m,
                "aic4": -2 * ll + 4 * m,
                "bic": -2 * ll + math.log(n) * m,
                "caic": -2 * ll + (math.log(n) + 1) * m,
                "mdl5": -2 * ll + 5 * m * math.log(n),
                "en": en,
            }
        )
