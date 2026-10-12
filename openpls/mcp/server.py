#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.
"""MCP server exposing the OpenPLS engine over stdio.

Run with ``openpls-mcp`` (after ``pip install openpls-engine[mcp]``) and point
an MCP client at it. Example client config (Claude Desktop / Code):

    {
      "mcpServers": {
        "openpls": { "command": "openpls-mcp" }
      }
    }

Everything runs locally: the tools read datasets from local file paths and
never upload data anywhere.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The MCP Python SDK renamed FastMCP -> MCPServer in v2; support both majors.
try:
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
except ModuleNotFoundError:
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        raise SystemExit(
            "The MCP server needs the 'mcp' package. Install the extra:\n"
            "    pip install openpls-engine[mcp]"
        ) from exc

from openpls.mcp import core

mcp = _Server("openpls-engine")


class LatentVariable(BaseModel):
    """One construct and the dataset columns that measure it."""

    name: str = Field(description="Construct name, e.g. 'Satisfaction'.")
    indicators: list[str] = Field(
        description="Dataset column names that measure this construct."
    )
    mode: Literal["A", "B"] = Field(
        default="A",
        description="Measurement mode: 'A' reflective, 'B' formative.",
    )


class StructuralPath(BaseModel):
    """A directed relationship between two constructs (source -> target)."""

    source: str = Field(description="Predictor construct name.")
    target: str = Field(description="Outcome construct name.")


@mcp.tool()
def describe_dataset(path: str) -> dict:
    """Inspect a local dataset before modelling.

    Returns the row count and, for every column, its name, whether it is
    numeric (usable as an indicator) and its non-null count. Use this to pick
    indicator columns for fit_pls_sem.
    """
    return core.describe_dataset(path)


@mcp.tool()
def fit_pls_sem(
    data_path: str,
    latent_variables: list[LatentVariable],
    structural_paths: list[StructuralPath],
    scheme: Literal["CENTROID", "PATH", "FACTORIAL", "NEWTON", "PCA"] = "CENTROID",
    scaled: bool = False,
    missing_codes: list[float] | None = None,
) -> dict:
    """Fit a PLS-SEM model to a local dataset and return the core results.

    Define each construct with its indicator columns and measurement mode, and
    the structural model as source -> target construct pairs. Returns R² per
    construct, path coefficients (estimate, p-value, f²), reliability
    (Cronbach's alpha, rho_a, rho_c, AVE), model fit (SRMR, d_ULS) and the
    goodness-of-fit index. Runs entirely locally; no data leaves the machine.
    """
    return core.fit_pls_sem(
        data_path,
        [lv.model_dump() for lv in latent_variables],
        [p.model_dump() for p in structural_paths],
        scheme=scheme,
        scaled=scaled,
        missing_codes=missing_codes,
    )


def main() -> None:
    """Console-script entry point: serve over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
