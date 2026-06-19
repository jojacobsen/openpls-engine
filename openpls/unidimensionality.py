#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
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

"""Backwards-compatibility shim for the pre-1.9.0 ``Unidimensionality`` name.

The class was renamed to :class:`openpls.reliability.Reliability` in 1.9.0
because Cronbach's alpha and Dillon-Goldstein rho_c measure reliability,
not unidimensionality. Import the new name; ``Unidimensionality`` will be
removed in a future release.
"""

import warnings

from openpls.reliability import Reliability


class Unidimensionality(Reliability):
    """Deprecated alias for :class:`openpls.reliability.Reliability`.

    Kept so external callers importing ``openpls.unidimensionality.Unidimensionality``
    continue to work; emits a :class:`DeprecationWarning` on construction.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "openpls.unidimensionality.Unidimensionality is deprecated; "
            "use openpls.reliability.Reliability instead. The metrics it "
            "computes (Cronbach's alpha, Dillon-Goldstein rho_c) measure "
            "reliability, not unidimensionality.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
