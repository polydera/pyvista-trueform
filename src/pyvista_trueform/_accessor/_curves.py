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
                             resolve_crossings=None,
                             resolve_self_crossings=None):
        """Intersection curves with ``other`` as a line-only PolyData.

        ``mode`` ("primitives" classifies shared edges/vertices and
        coplanar contacts, "sos" perturbs every contact into a crossing),
        ``tolerance`` (world-coordinate placement distance, 0 = exact),
        ``resolve_crossings`` (crossings between different contours on the
        same face), and ``resolve_self_crossings`` (self-crossings within
        one contour). When omitted, trueform's defaults apply — see
        :func:`trueform.intersection_curves`.
        """
        return curves_to_pyvista(
            tf.intersection_curves(
                self.to_mesh(), _operand_mesh(other),
                **_forwarded(mode=mode, tolerance=tolerance,
                            resolve_crossings=resolve_crossings,
                            resolve_self_crossings=resolve_self_crossings)))

    def self_intersection_curves(self, *, mode=None, tolerance=None,
                                  resolve_crossings=None,
                                  resolve_self_crossings=None):
        """This mesh's self-intersection curves as a line-only PolyData.

        See :meth:`intersection_curves` for the same keyword arguments
        (``mode``, ``tolerance``, ``resolve_crossings``,
        ``resolve_self_crossings``) — trueform defaults both crossing
        options to True here, since a single contour's self-crossings are
        the whole point. See :func:`trueform.self_intersection_curves`.
        """
        return curves_to_pyvista(
            tf.self_intersection_curves(
                self.to_mesh(),
                **_forwarded(mode=mode, tolerance=tolerance,
                            resolve_crossings=resolve_crossings,
                            resolve_self_crossings=resolve_self_crossings)))
