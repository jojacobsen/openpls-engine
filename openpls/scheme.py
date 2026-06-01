#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
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

from enum import Enum

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

import openpls.util as util


class _CentroidInnerWeightCalculator(util.Value):

    def __init__(self):
        super().__init__("C")

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return np.sign(np.corrcoef(y, rowvar=False) * (path + path.transpose()))


class _FactorialInnerWeightCalculator(util.Value):

    def __init__(self):
        super().__init__("F")

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return np.cov(y, rowvar=False) * (path + path.transpose())


class _PathInnerWeightCalculator(util.Value):

    def __init__(self):
        super().__init__("P")

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        E = path.values.astype(np.float64)
        for i in range(E.shape[0]):
            follow = path.iloc[i, :] == 1
            if path.iloc[i, :].sum() > 0:
                E[follow, i] = sm.OLS(y[:, i], y[:, follow]).fit().params
            predec = path.iloc[:, i] == 1
            if path.iloc[:, i].sum() > 0:
                E[predec, i] = np.corrcoef(np.column_stack((y[:, predec], y[:, i])), rowvar=False)[:,-1][:-1]
        return E


class _NewtonInnerWeightCalculator(util.Value):
    """Quasi-Newton (BFGS) joint optimization of inner weights.

    For each latent variable Y_j with neighborhood N(j) = predecessors ∪
    successors, fits a single joint weight vector e_j by minimizing the
    least-squares loss ||Y_j - sum_{k ∈ N(j)} e_jk * Y_k||^2 via BFGS
    (the classical quasi-Newton method, using a positive-definite secant
    approximation of the Hessian).

    This differs from the PATH scheme, which treats predecessors and
    successors asymmetrically: predecessors get OLS regression
    coefficients while successors get bare correlations. The NEWTON
    scheme handles all neighbors uniformly under one joint optimization
    objective, giving a single optimization-coherent weight vector per
    LV. BFGS is initialized from the analytical OLS solution and
    terminates when the gradient norm drops below 1e-8, so for the
    convex quadratic objective the result is numerically identical to
    joint OLS while keeping the implementation extensible to
    regularized or non-quadratic future variants.

    References
    ----------
    - Nocedal, J., & Wright, S. J. (2006). Numerical Optimization
      (2nd ed.). Chapter 6 (Quasi-Newton Methods).
    - Tenenhaus, M., Esposito Vinzi, V., Chatelin, Y.-M., & Lauro, C.
      (2005). PLS path modeling. Computational Statistics & Data
      Analysis, 48(1), 159-205.
    """

    def __init__(self):
        super().__init__("N")

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        n_lvs = path.shape[0]
        E = np.zeros((n_lvs, n_lvs), dtype=np.float64)
        adj = (path.values + path.values.T).astype(bool)
        for j in range(n_lvs):
            neighbors = np.where(adj[j])[0]
            if neighbors.size == 0:
                continue
            Y_n = y[:, neighbors]
            y_j = y[:, j]
            # Initialize from analytical OLS (warm-start BFGS).
            init, *_ = np.linalg.lstsq(Y_n, y_j, rcond=None)

            def loss(e, Y_n=Y_n, y_j=y_j):
                resid = y_j - Y_n @ e
                return 0.5 * float(resid @ resid)

            def grad(e, Y_n=Y_n, y_j=y_j):
                return -Y_n.T @ (y_j - Y_n @ e)

            result = minimize(
                loss, init, jac=grad, method="BFGS",
                options={"gtol": 1e-8, "maxiter": 200},
            )
            E[j, neighbors] = result.x
        # Symmetrize the matrix the same way Centroid/Factorial/Path schemes
        # do implicitly via the (path + path.T) adjacency: each connected pair
        # (j, k) gets a weight on both sides. NEWTON above already populates
        # both E[j, k] and E[k, j] independently via per-LV optimization, so
        # no extra symmetrization step is needed.
        return E


class _PCAInnerWeightCalculator(util.Value):
    """Lohmöller's PCA inner-weighting scheme.

    For each latent variable Y_j with neighborhood N(j), the inner
    weights are the components of the first principal direction of the
    neighbor-score matrix Y_N(j). Equivalently, the inner estimate
    Z_j = Y_N(j) @ e_j is (up to sign) the first principal component
    of the neighboring LVs, the linear combination that captures the
    most joint variance among them.

    Differs from CENTROID (which uses sign(cor)) and FACTORIAL (which
    uses cor itself) by treating the neighbor weights as a joint
    multivariate direction, rather than as separate pairwise quantities.
    Each PC direction is sign-flipped to correlate positively with the
    central LV Y_j, ensuring the resulting inner estimate has the same
    orientation as Y_j and the outer Wold loop stays sign-consistent.

    References
    ----------
    - Lohmöller, J.-B. (1989). Latent Variable Path Modeling with
      Partial Least Squares. Physica-Verlag, Section 2.4.2.
    """

    def __init__(self):
        super().__init__("L")

    def calculate(self, path: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        n_lvs = path.shape[0]
        E = np.zeros((n_lvs, n_lvs), dtype=np.float64)
        adj = (path.values + path.values.T).astype(bool)
        for j in range(n_lvs):
            neighbors = np.where(adj[j])[0]
            if neighbors.size == 0:
                continue
            Y_n = y[:, neighbors]
            if neighbors.size == 1:
                # only one neighbor: PC direction is trivially [1]
                e_j = np.array([1.0])
            else:
                # first principal direction via SVD of centered Y_n
                _, _, vt = np.linalg.svd(Y_n - Y_n.mean(axis=0), full_matrices=False)
                e_j = vt[0]
            # sign-flip so the PC correlates positively with Y_j
            pc_score = Y_n @ e_j
            if np.corrcoef(pc_score, y[:, j])[0, 1] < 0:
                e_j = -e_j
            E[j, neighbors] = e_j
        return E


class Scheme(Enum):
    """
    The scheme to use to calculate inner weights.
    """
    CENTROID = _CentroidInnerWeightCalculator()
    PATH = _PathInnerWeightCalculator()
    FACTORIAL = _FactorialInnerWeightCalculator()
    NEWTON = _NewtonInnerWeightCalculator()
    PCA = _PCAInnerWeightCalculator()
