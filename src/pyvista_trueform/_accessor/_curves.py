"""
Intersection curves on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista
from .._forward import _forwarded
from . import _operand_mesh


class _CurvesMixin:

    def intersection_curves(self, other, *, mode=None, tolerance=None,
                             within=None):
        """Intersection curves with ``other`` as a line-only PolyData.

        ``mode`` ("primitives" classifies shared edges/vertices and
        coplanar contacts, "sos" perturbs every contact into a crossing),
        ``tolerance`` (world-coordinate placement distance, 0 = exact),
        and ``within`` (also intersect each mesh with itself, so its own
        self-crossings resolve in the arrangement these curves are read
        from — what is emitted stays the cross-mesh seams). Crossings
        between contours resolve unconditionally; there is no flag for
        them. When omitted, trueform's defaults apply — see
        :func:`trueform.intersection_curves`.
        """
        return curves_to_pyvista(
            tf.intersection_curves(
                self.to_mesh(), _operand_mesh(other),
                **_forwarded(mode=mode, tolerance=tolerance,
                            within=within)))

    def self_intersection_curves(self, *, mode=None, tolerance=None):
        """This mesh's self-intersection curves as a line-only PolyData.

        See :meth:`intersection_curves` for ``mode`` and ``tolerance``. A
        one-form build implies ``within``, so this mesh meeting itself is
        what the curves are read from and there is no keyword for it. See
        :func:`trueform.self_intersection_curves`.
        """
        return curves_to_pyvista(
            tf.self_intersection_curves(
                self.to_mesh(),
                **_forwarded(mode=mode, tolerance=tolerance)))
