#!/usr/bin/python3
#
# Copyright (C) 2026 OpenPLS contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version. See <http://www.gnu.org/licenses/>.
"""Model Context Protocol (MCP) server for the OpenPLS engine.

Exposes the engine as local tools an AI assistant (Claude, Cursor, …) can
call over stdio: inspect a dataset and fit a PLS-SEM model, entirely on the
user's machine with no cloud, no account and no data upload.

The compute logic lives in :mod:`openpls.mcp.core` and has no dependency on
the MCP SDK, so it is importable and testable without the ``[mcp]`` extra.
The thin protocol wrapper lives in :mod:`openpls.mcp.server` and requires
``pip install openpls-engine[mcp]``.
"""
