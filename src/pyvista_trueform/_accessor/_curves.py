"""
Intersection curves on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista
from . import _operand_mesh


class _CurvesMixin:

    def intersection_curves(self, other, **kwargs):
        """Intersection curves with ``other`` as a line-only PolyData.

        Keyword arguments forward to
        :func:`trueform.intersection_curves`: ``mode`` ("primitives"
        classifies shared edges/vertices and coplanar contacts, "sos"
        perturbs every contact into a crossing), ``tolerance`` (world-
        coordinate placement distance, 0 = exact), ``resolve_crossings``
        (crossings between different contours on the same face), and
        ``resolve_self_crossings`` (self-crossings within one contour).
        """
        return curves_to_pyvista(
            tf.intersection_curves(self.to_mesh(), _operand_mesh(other),
                                   **kwargs))

    def self_intersection_curves(self, **kwargs):
        """This mesh's self-intersection curves as a line-only PolyData.

        See :func:`trueform.self_intersection_curves` for the same keyword
        arguments as :meth:`intersection_curves` (``mode``, ``tolerance``,
        ``resolve_crossings``, ``resolve_self_crossings`` — both default
        True here, since a single contour's self-crossings are the whole
        point).
        """
        return curves_to_pyvista(
            tf.self_intersection_curves(self.to_mesh(), **kwargs))
