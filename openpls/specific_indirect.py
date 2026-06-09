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

"""Specific indirect effects for PLS-SEM path models.

Implements the chain-product approach used in mediation analysis
(Zhao, Lynch & Chen 2010; Nitzl, Roldan & Cepeda 2016). Each indirect
path source -> M1 -> ... -> target carries a *specific indirect effect*
equal to the product of the path coefficients along the chain. The total
indirect effect from source to target is the sum of all such products.

The R package seminr exposes this as ``specific_effect_significance``;
this module is the Python equivalent and integrates with both the point
estimate (:class:`~openpls.plspm.Plspm`) and the bootstrap inference
layer (:class:`~openpls.bootstrap.Bootstrap`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _children(path: pd.DataFrame, source: str) -> list[str]:
    """Direct successors of ``source`` in the structural DAG.

    The path matrix has rows = targets, columns = sources; a 1 in
    ``path.loc[target, source]`` means source -> target.
    """
    col = path.loc[:, source]
    return list(col[col == 1].index)


def enumerate_chains(
    path: pd.DataFrame,
    source: str,
    target: str,
) -> list[list[str]]:
    """Enumerate all simple directed chains from ``source`` to ``target``.

    Returns a list of chains. Each chain is a list ``[source, m1, ..., target]``
    of length >= 3 (i.e. excludes the direct edge source -> target). The
    structural model is assumed acyclic; cycle-safety is enforced by
    tracking visited nodes per DFS branch.

    Args:
        path: structural-model path matrix (rows = targets, cols = sources;
            ``path.loc[t, s] == 1`` means s -> t).
        source: source LV name.
        target: target LV name.

    Returns:
        list of chains, possibly empty.

    Raises:
        KeyError: if ``source`` or ``target`` is not a node in ``path``.
        ValueError: if ``source == target``.
    """
    if source == target:
        raise ValueError("source and target must differ")
    nodes = set(path.index) | set(path.columns)
    if source not in nodes:
        raise KeyError(f"unknown LV: {source!r}")
    if target not in nodes:
        raise KeyError(f"unknown LV: {target!r}")

    chains: list[list[str]] = []

    def _dfs(node: str, trail: list[str], visited: set[str]) -> None:
        for child in _children(path, node):
            if child == target:
                if len(trail) >= 2:
                    chains.append(trail + [child])
                continue
            if child in visited:
                continue
            visited.add(child)
            _dfs(child, trail + [child], visited)
            visited.remove(child)

    _dfs(source, [source], {source})
    return chains


def _chain_label(chain: list[str]) -> str:
    return " -> ".join(chain)


def _chain_edge_keys(chain: list[str]) -> list[str]:
    return [f"{chain[i]} -> {chain[i + 1]}" for i in range(len(chain) - 1)]


def _resolve_chain(
    path: pd.DataFrame,
    source: str,
    target: str,
    through: list[str] | None,
) -> list[list[str]]:
    """Validate ``through`` or enumerate all chains."""
    if through is None:
        chains = enumerate_chains(path, source, target)
        if not chains:
            raise ValueError(
                f"no indirect path found from {source!r} to {target!r}"
            )
        return chains
    if not through:
        raise ValueError("through must be non-empty (or None to auto-enumerate)")
    chain = [source, *through, target]
    if len(set(chain)) != len(chain):
        raise ValueError(f"chain contains repeated LVs: {chain}")
    for parent, child in zip(chain[:-1], chain[1:]):
        if path.loc[child, parent] != 1:
            raise ValueError(
                f"no direct edge {parent!r} -> {child!r} in the structural model"
            )
    return [chain]


def specific_indirect_point(
    path_coefficients: pd.DataFrame,
    path: pd.DataFrame,
    source: str,
    target: str,
    through: list[str] | None = None,
) -> pd.DataFrame:
    """Point estimates of specific indirect effects.

    Each row reports one source -> ... -> target chain together with the
    product of its path coefficients. With ``through=None`` every indirect
    chain in the structural model is returned; with ``through`` set, only
    that single chain is returned.

    Returns:
        DataFrame indexed by chain label (e.g. ``"A -> M -> Y"``) with
        columns ``from``, ``to``, ``via`` (tuple of intermediates), and
        ``estimate``.
    """
    chains = _resolve_chain(path, source, target, through)
    rows = []
    for chain in chains:
        prod = 1.0
        for parent, child in zip(chain[:-1], chain[1:]):
            prod *= float(path_coefficients.loc[child, parent])
        rows.append({
            "from": chain[0],
            "to": chain[-1],
            "via": tuple(chain[1:-1]),
            "estimate": prod,
        })
    return pd.DataFrame(rows, index=[_chain_label(c) for c in chains])


def specific_indirect_bootstrap(
    raw_paths: pd.DataFrame,
    path_coefficients: pd.DataFrame,
    path: pd.DataFrame,
    source: str,
    target: str,
    through: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Bootstrap inference for specific indirect effects.

    For every chain source -> ... -> target, multiplies the per-iteration
    path coefficients (columns of ``raw_paths`` keyed as ``"A -> B"``)
    along the chain and summarises the resulting empirical distribution.

    Args:
        raw_paths: bootstrap iterations of *direct* path coefficients,
            with columns labelled ``"A -> B"`` (one row per iteration).
        path_coefficients: point-estimate path coefficients (target rows,
            source columns).
        path: structural path matrix.
        source: source LV.
        target: target LV.
        through: explicit chain or ``None`` to enumerate.
        alpha: two-sided percentile-CI level (default 0.05 → 95% CI).

    Returns:
        DataFrame indexed by chain label with columns ``from``, ``to``,
        ``via``, ``original``, ``mean``, ``std.error``, ``perc.lower``,
        ``perc.upper``, ``t stat.``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    chains = _resolve_chain(path, source, target, through)

    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0

    rows = []
    for chain in chains:
        edge_keys = _chain_edge_keys(chain)
        missing = [k for k in edge_keys if k not in raw_paths.columns]
        if missing:
            raise KeyError(
                f"raw bootstrap paths missing edges {missing!r}; "
                "rerun Plspm with bootstrap=True"
            )
        products = raw_paths.loc[:, edge_keys].prod(axis=1).astype(float)
        original = 1.0
        for parent, child in zip(chain[:-1], chain[1:]):
            original *= float(path_coefficients.loc[child, parent])
        std = float(products.std(ddof=1))
        rows.append({
            "from": chain[0],
            "to": chain[-1],
            "via": tuple(chain[1:-1]),
            "original": original,
            "mean": float(products.mean()),
            "std.error": std,
            "perc.lower": float(products.quantile(lower_q)),
            "perc.upper": float(products.quantile(upper_q)),
            "t stat.": original / std if std > 0 else np.nan,
        })
    return pd.DataFrame(rows, index=[_chain_label(c) for c in chains])
