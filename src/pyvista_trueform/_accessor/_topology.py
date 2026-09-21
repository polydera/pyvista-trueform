"""
Topology reads on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf

from .._conversion import curves_to_pyvista, to_pyvista
from .._forward import _forwarded


def _edge_lines(edges, points):
    """``(N, 2)`` vertex-id pairs as one-segment line cells over ``points``."""
    offsets = np.arange(0, 2 * len(edges) + 2, 2, dtype=edges.dtype)
    return curves_to_pyvista(
        tf.OffsetBlockedArray(offsets, edges.reshape(-1)), points)


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

    def non_manifold_edges(self):
        """Every edge shared by more than two faces, as a line-only PolyData.

        One line cell per ``(N, 2)`` edge of
        :func:`trueform.non_manifold_edges`, its two ids naming this
        dataset's own points — the result carries the full point array, so
        cell ids read straight back into the dataset. An empty PolyData
        when the mesh is manifold.
        """
        edges = tf.non_manifold_edges(self.to_mesh())
        if len(edges) == 0:
            return pv.PolyData()
        return _edge_lines(edges, self.to_mesh().points)

    def non_manifold_paths(self):
        """The non-manifold edges assembled into polylines.

        The edges of :meth:`non_manifold_edges` connected through
        :func:`trueform.connect_edges_to_paths`, one line cell per
        polyline, ids naming this dataset's own points. An empty PolyData
        when the mesh is manifold.
        """
        edges = tf.non_manifold_edges(self.to_mesh())
        if len(edges) == 0:
            return pv.PolyData()
        return curves_to_pyvista(tf.connect_edges_to_paths(edges),
                                 self.to_mesh().points)

    def non_manifold_vertices(self):
        """Every vertex whose faces are not one fan, as an ascending
        ``(N,)`` array of this dataset's own point ids.

        An edge at the vertex carries three or more faces, or its faces
        fall into several fans meeting at the vertex alone (a bowtie).
        Winding does not enter the verdict, and a vertex no face names is
        not reported. Empty when the mesh is manifold. See
        :func:`trueform.non_manifold_vertices`.
        """
        return tf.non_manifold_vertices(self.to_mesh())

    def boundary_edges(self):
        """Every edge belonging to exactly one face, as a line-only PolyData.

        One line cell per ``(N, 2)`` edge of
        :func:`trueform.boundary_edges`, its two ids naming this dataset's
        own points — the result carries the full point array, so cell ids
        read straight back into the dataset. An empty PolyData when the
        mesh is closed.
        """
        edges = tf.boundary_edges(self.to_mesh())
        if len(edges) == 0:
            return pv.PolyData()
        return _edge_lines(edges, self.to_mesh().points)

    def boundary_paths(self):
        """The boundary edges assembled into loops, ids naming this
        dataset's own points.

        One line cell per loop of :func:`trueform.boundary_paths`, closed
        by repeating its first id. An empty PolyData when the mesh is
        closed. :meth:`boundary_curves` answers the same loops over a
        compacted point set of its own instead.
        """
        paths = tf.boundary_paths(self.to_mesh())
        if len(paths.data) == 0:
            return pv.PolyData()
        return curves_to_pyvista(paths, self.to_mesh().points)

    def boundary_rims(self):
        """The boundary as rims: the vertices each one walks, the face
        carrying each of its edges, and whether it closes.

        Returns ``(vertices, faces, closed)`` — two
        :class:`trueform.OffsetBlockedArray` of one block per rim, passed
        through untouched, and a ``(R,)`` int8 array nonzero where rim
        ``i``'s last edge runs back to its first vertex. Rim edge ``k``
        runs from vertex ``k`` to vertex ``k + 1`` and is carried by face
        ``k`` alone, so a closed rim of ``n`` vertices has ``n`` edges and
        an open one ``n - 1``; the ids name this dataset's own points and
        cells. A rim ends where the boundary stops passing straight
        through, so a pinch splits it. :meth:`boundary_paths` draws the
        same boundary as a line-only PolyData instead. See
        :func:`trueform.boundary_rims`.
        """
        return tf.boundary_rims(self.to_mesh())
