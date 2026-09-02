"""
Scalar-field cuts on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista


class _IsoMixin:

    def isocontours(self, scalars, threshold):
        """Isocontour curves of a per-vertex scalar field as line PolyData.

        ``scalars`` is a ``point_data`` array name or an array;
        ``threshold`` a value or an array of values. See
        :func:`trueform.isocontours`.
        """
        return curves_to_pyvista(
            tf.isocontours(self.to_mesh(), self._scalar_field(scalars),
                           threshold))

    def isobands(self, scalars, cut_values, *, selected_bands=None,
                 return_curves=False):
        """The mesh recut into scalar-field bands.

        The band of every face rides as ``trueform_labels``, its source face
        as ``trueform_face_labels``. See :func:`trueform.isobands`.
        """
        result = tf.isobands(self.to_mesh(), self._scalar_field(scalars),
                             cut_values, selected_bands=selected_bands,
                             return_curves=return_curves)
        if return_curves:
            mesh, labels, face_labels, curves = result
            return (self._labeled(mesh, labels, face_labels),
                    curves_to_pyvista(curves))
        mesh, labels, face_labels = result
        return self._labeled(mesh, labels, face_labels)
