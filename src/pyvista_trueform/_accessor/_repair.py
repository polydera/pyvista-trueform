"""
Self-intersection repair and mesh processing on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista, to_pyvista


class _RepairMixin:

    def polygon_arrangements(self, *, return_curves=False, **kwargs):
        """The mesh split at its own self-intersection curves.

        Per-face provenance rides as ``trueform_face_labels``. See
        :func:`trueform.polygon_arrangements` for keyword arguments.
        """
        result = tf.polygon_arrangements(
            self.to_mesh(), return_curves=return_curves, **kwargs)
        if return_curves:
            mesh, face_labels, curves = result
            return (self._labeled(mesh, face_labels=face_labels),
                    curves_to_pyvista(curves))
        mesh, face_labels = result
        return self._labeled(mesh, face_labels=face_labels)

    def outer_shell(self):
        """Repair to the outer shell: the boundary of the union of
        everything the mesh encloses, free of self-intersections.

        See :func:`trueform.outer_shell`.
        """
        return to_pyvista(tf.outer_shell(self.to_mesh()))

    def cleaned(self, tolerance=None, **kwargs):
        """Duplicate vertices and degenerate faces removed.

        See :func:`trueform.cleaned` for keyword arguments.
        """
        return to_pyvista(tf.cleaned(self.to_mesh(), tolerance, **kwargs))

    def triangulated(self):
        """Every face triangulated on its own boundary, shared edges one
        identity in both faces.

        See :func:`trueform.triangulated`.
        """
        return to_pyvista(tf.triangulated(self.to_mesh()))
