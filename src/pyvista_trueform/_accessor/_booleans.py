"""
Boolean operations on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista
from . import _operand_mesh


class _BooleansMixin:

    def _boolean(self, operation, other, return_curves):
        result = operation(self.to_mesh(), _operand_mesh(other),
                           return_curves=return_curves)
        if return_curves:
            mesh, labels, face_labels, curves = result
            return (self._labeled(mesh, labels, face_labels),
                    curves_to_pyvista(curves))
        mesh, labels, face_labels = result
        return self._labeled(mesh, labels, face_labels)

    def union(self, other, *, return_curves=False):
        """Boolean union with ``other`` (PolyData or trueform Mesh).

        With ``return_curves=True`` also returns the intersection curves as
        a second, line-only PolyData. See :func:`trueform.boolean_union`.
        """
        return self._boolean(tf.boolean_union, other, return_curves)

    def intersection(self, other, *, return_curves=False):
        """Boolean intersection with ``other`` (PolyData or trueform Mesh).

        See :func:`trueform.boolean_intersection`.
        """
        return self._boolean(tf.boolean_intersection, other, return_curves)

    def difference(self, other, *, return_curves=False):
        """Boolean difference: this mesh minus ``other``.

        The other direction is the other dataset's accessor:
        ``other.trueform.difference(this)``. See
        :func:`trueform.boolean_difference`.
        """
        return self._boolean(tf.boolean_difference, other, return_curves)
