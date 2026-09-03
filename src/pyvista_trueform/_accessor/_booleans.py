"""
Boolean operations on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista
from .._forward import _forwarded
from . import _operand_mesh


class _BooleansMixin:

    def _boolean(self, operation, other, return_curves, sheets):
        result = operation(self.to_mesh(), _operand_mesh(other),
                           return_curves=return_curves,
                           **_forwarded(sheets=sheets))
        if return_curves:
            mesh, labels, face_labels, curves = result
            return (self._labeled(mesh, labels, face_labels),
                    curves_to_pyvista(curves))
        mesh, labels, face_labels = result
        return self._labeled(mesh, labels, face_labels)

    def union(self, other, *, return_curves=False, sheets=None):
        """Boolean union with ``other`` (PolyData or trueform Mesh).

        With ``return_curves=True`` also returns the intersection curves as
        a second, line-only PolyData. ``sheets`` names operand indices
        (``0``/``1``) declared as oriented separators that bound no volume.
        See :func:`trueform.boolean_union`.
        """
        return self._boolean(tf.boolean_union, other, return_curves, sheets)

    def intersection(self, other, *, return_curves=False, sheets=None):
        """Boolean intersection with ``other`` (PolyData or trueform Mesh).

        With ``return_curves=True`` also returns the intersection curves as
        a second, line-only PolyData. ``sheets`` names operand indices
        (``0``/``1``) declared as oriented separators that bound no volume.
        See :func:`trueform.boolean_intersection`.
        """
        return self._boolean(tf.boolean_intersection, other, return_curves,
                             sheets)

    def difference(self, other, *, return_curves=False, sheets=None):
        """Boolean difference: this mesh minus ``other``.

        The other direction is the other dataset's accessor:
        ``other.trueform.difference(this)``. With ``return_curves=True``
        also returns the intersection curves as a second, line-only
        PolyData. ``sheets`` names operand indices (``0``/``1``) declared
        as oriented separators that bound no volume. See
        :func:`trueform.boolean_difference`.
        """
        return self._boolean(tf.boolean_difference, other, return_curves,
                             sheets)
