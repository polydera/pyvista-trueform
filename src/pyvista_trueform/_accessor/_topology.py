"""
Topology reads on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv
import trueform as tf

from .._conversion import to_pyvista
from .._forward import _forwarded


class _TopologyMixin:

    def connected_components(self, *, expected_number_of_components=None):
        """Label every face with its manifold-edge-connected component.

        Returns ``(n, labels)`` — the component count and a ``(n_faces,)``
        int32 array, faces sharing a manifold edge sharing a label. The
        adjacency is the cached mesh's own lazily built manifold edge
        link, so repeated topology reads amortize.
        ``expected_number_of_components`` is
        :func:`trueform.label_connected_components`'s planning hint,
        forwarded verbatim; trueform's default applies when omitted.
        """
        return tf.label_connected_components(
            self.to_mesh().manifold_edge_link,
            **_forwarded(
                expected_number_of_components=expected_number_of_components))

    def split_components(self):
        """Every manifold-edge-connected component as its own block.

        Block ``k`` is component ``k`` of :meth:`connected_components` —
        a fresh PolyData named ``str(k)``, its points reindexed to the
        ones the component uses. Returns a :class:`pyvista.MultiBlock`
        of ``n`` blocks. See :func:`trueform.split_into_components`.
        """
        mesh = self.to_mesh()
        _, labels = tf.label_connected_components(mesh.manifold_edge_link)
        components, component_labels = tf.split_into_components(mesh, labels)
        blocks = pv.MultiBlock()
        for component, label in zip(components, component_labels):
            blocks.append(to_pyvista(component), str(label))
        return blocks
