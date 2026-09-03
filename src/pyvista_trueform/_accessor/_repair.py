"""
Self-intersection repair and mesh processing on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista, domains_to_pyvista, to_pyvista


class _RepairMixin:

    def domains(self, expr=None, **kwargs):
        """This mesh's self-decomposition into volumetric domains.

        The mesh is read through its own self arrangement — a
        :class:`trueform.CsgGraph` of one operand — whose overlap pockets
        classify into domains; block ``k`` is domain ``ids[k]``, named
        ``str(ids[k])``. See :meth:`trueform.CsgGraph.domains` for keyword
        arguments (``selection``, ``exclude_outer_shell``,
        ``ignore_open_fragments``, ``return_source_ids``,
        ``return_index_map``). With ``return_source_ids=True`` also returns
        per-cell face provenance as two :class:`trueform.OffsetBlockedArray`,
        parallel to the block list, passed through untouched; with
        ``return_index_map=True`` returns ``(multiblock, index_map)``, the
        :class:`trueform.DomainsIndexMap` passed through untouched.

        Returns
        -------
        pyvista.MultiBlock
        """
        graph = tf.CsgGraph([self.to_mesh()])
        result = graph.domains(expr, **kwargs)
        if kwargs.get("return_index_map"):
            cells, ids, index_map = result
            return domains_to_pyvista(cells, ids), index_map
        if kwargs.get("return_source_ids"):
            cells, ids, tag_blocks, face_blocks = result
            return domains_to_pyvista(cells, ids), tag_blocks, face_blocks
        cells, ids = result
        return domains_to_pyvista(cells, ids)

    def polygon_arrangements(self, *, return_curves=False, **kwargs):
        """The mesh split at its own self-intersection curves.

        Per-face provenance rides as ``trueform_face_labels``. See
        :func:`trueform.polygon_arrangements` for the remaining keyword
        arguments: ``mode`` ("primitives" or "sos"), ``tolerance``,
        ``resolve_crossings``, ``resolve_self_crossings``, and
        ``triangulation`` ("cdt" or "refined_cdt").
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

        ``tolerance`` merges vertices within that world-coordinate
        distance (``None`` keeps exact-duplicate merging only). With
        ``return_index_map=True`` also returns the face and point index
        maps, each a ``(f, kept_ids)`` pair, exactly as
        :func:`trueform.cleaned` returns them for a mesh — passed through
        untouched. See :func:`trueform.cleaned` for the remaining keyword
        arguments (``remove_duplicate_primitives``,
        ``remove_unreferenced_points``).
        """
        result = tf.cleaned(self.to_mesh(), tolerance, **kwargs)
        if kwargs.get("return_index_map"):
            mesh, face_map, point_map = result
            return to_pyvista(mesh), face_map, point_map
        return to_pyvista(result)

    def triangulated(self):
        """Every face triangulated on its own boundary, shared edges one
        identity in both faces.

        See :func:`trueform.triangulated`.
        """
        return to_pyvista(tf.triangulated(self.to_mesh()))
