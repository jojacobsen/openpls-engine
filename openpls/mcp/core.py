#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.
"""Pure compute helpers behind the OpenPLS MCP tools.

No MCP-SDK import here on purpose: this module is importable and unit-testable
with just the engine's runtime dependencies. The MCP wrapper in
:mod:`openpls.mcp.server` converts tool arguments to plain structures and calls
these functions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

import openpls.config as c
from openpls import Plspm
from openpls.mode import Mode
from openpls.scheme import Scheme


def _num(value: Any) -> float | None:
    """Coerce to float, mapping NaN/inf to None so the result is JSON-safe."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def read_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from a local path (CSV/TSV/Excel) into a DataFrame."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"Dataset not found: {path}")
    ext = p.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(p)
    if ext == ".tsv":
        return pd.read_csv(p, sep="\t")
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ValueError(
                "Reading Excel files needs an extra engine (openpyxl for .xlsx, "
                "xlrd for .xls). Install it or convert the file to CSV."
            ) from exc
    raise ValueError(f"Unsupported dataset format '{ext}'. Use CSV, TSV or Excel.")


def describe_dataset(path: str) -> dict[str, Any]:
    """Return row count and per-column name/type for a dataset.

    Lets an assistant see which columns exist (and which are numeric, i.e.
    usable as indicators) before specifying a model.
    """
    df = read_dataset(path)
    columns = []
    for name in df.columns:
        numeric = pd.api.types.is_numeric_dtype(df[name])
        columns.append(
            {
                "name": str(name),
                "type": "numeric" if numeric else "categorical",
                "non_null": int(df[name].notna().sum()),
            }
        )
    return {"path": str(path), "rows": int(len(df)), "columns": columns}


def fit_pls_sem(
    data_path: str,
    latent_variables: list[dict[str, Any]],
    structural_paths: list[dict[str, Any]],
    scheme: str = "CENTROID",
    scaled: bool = False,
    missing_codes: list[float] | None = None,
) -> dict[str, Any]:
    """Fit a PLS-SEM model and return the core results.

    Args:
        data_path: local path to the dataset (CSV/TSV/Excel).
        latent_variables: list of ``{"name", "indicators": [...], "mode": "A"|"B"}``;
            mode A is reflective, B is formative (default A).
        structural_paths: list of ``{"source", "target"}`` LV-name pairs.
        scheme: inner-weighting scheme (CENTROID, PATH, FACTORIAL, NEWTON, PCA).
        scaled: standardise indicators before fitting.
        missing_codes: numeric sentinels (e.g. ``[-99, -999]``) to treat as missing.

    Returns a JSON-serialisable dict with r_squared per LV, path coefficients
    (estimate, p-value, f²), reliability (alpha, rho_a, rho_c, AVE), model fit
    (SRMR, d_ULS) and the goodness-of-fit index.
    """
    df = read_dataset(data_path)
    if missing_codes:
        df = df.replace(list(missing_codes), float("nan"))

    scheme_key = (scheme or "CENTROID").upper()
    schemes = {s.name: s for s in Scheme}
    if scheme_key not in schemes:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose one of {sorted(schemes)}.")
    modes = {"A": Mode.A, "B": Mode.B}

    if not latent_variables:
        raise ValueError("At least one latent variable is required.")
    if not structural_paths:
        raise ValueError("At least one structural path is required.")

    # Validate before touching the engine so errors are actionable.
    lv_names: list[str] = []
    for lv in latent_variables:
        name = lv.get("name")
        indicators = lv.get("indicators") or []
        mode = str(lv.get("mode", "A")).upper()
        if not name:
            raise ValueError("Each latent variable needs a 'name'.")
        if mode not in modes:
            raise ValueError(f"LV '{name}': mode must be 'A' (reflective) or 'B' (formative).")
        if not indicators:
            raise ValueError(f"LV '{name}' has no indicators.")
        missing = [col for col in indicators if col not in df.columns]
        if missing:
            raise ValueError(f"LV '{name}': indicators not in dataset: {missing}.")
        lv_names.append(name)

    lv_set = set(lv_names)
    for p in structural_paths:
        src, tgt = p.get("source"), p.get("target")
        if src not in lv_set or tgt not in lv_set:
            raise ValueError(
                f"Path {src} -> {tgt} references an unknown LV "
                f"(known: {sorted(lv_set)})."
            )

    # Build the structural model, grouping targets per source.
    structure = c.Structure()
    targets_by_source: dict[str, list[str]] = {}
    for p in structural_paths:
        targets_by_source.setdefault(p["source"], []).append(p["target"])
    for src, targets in targets_by_source.items():
        structure.add_path([src], targets)

    config = c.Config(structure.path(), scaled=scaled)
    for lv in latent_variables:
        mvs = [c.MV(col) for col in lv["indicators"]]
        config.add_lv(lv["name"], modes[str(lv.get("mode", "A")).upper()], *mvs)

    fit = Plspm(df, config, schemes[scheme_key])
    return _extract_results(fit)


def _extract_results(fit: Plspm) -> dict[str, Any]:
    report = fit.report()

    inner = fit.inner_summary()
    r_squared = [
        {
            "lv": str(lv),
            "type": str(row.get("type", "")),
            "r_squared": _num(row.get("r_squared")),
            "r_squared_adj": _num(row.get("r_squared_adj")),
        }
        for lv, row in inner.iterrows()
    ]

    paths = [
        {
            "source": str(rec["from"]),
            "target": str(rec["to"]),
            "estimate": _num(rec.get("estimate")),
            "p_value": _num(rec.get("p_value")),
            "t": _num(rec.get("t")),
            "f_squared": _num(rec.get("f_squared")),
        }
        for rec in report.paths().to_dict("records")
    ]

    rel = report.reliability()
    reliability = [
        {
            "lv": str(lv),
            "cronbach_alpha": _num(row.get("cronbach_alpha")),
            "rho_a": _num(row.get("rho_a")),
            "rho_c": _num(row.get("rho_c")),
            "ave": _num(row.get("ave")),
        }
        for lv, row in rel.iterrows()
    ]

    mf = fit.model_fit()
    model_fit = {"srmr": _num(mf.srmr()), "d_uls": _num(mf.d_uls())}

    return {
        "goodness_of_fit": _num(fit.goodness_of_fit()),
        "model_fit": model_fit,
        "r_squared": r_squared,
        "path_coefficients": paths,
        "reliability": reliability,
    }
