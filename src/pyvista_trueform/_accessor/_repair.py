"""
Self-intersection repair and mesh processing on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._conversion import curves_to_pyvista, domains_to_pyvista, to_pyvista
from .._forward import _forwarded


class _RepairMixin:

    def domains(self, expr=None, *, selection=None, exclude_outer_shell=None,
               ignore_open_fragments=None, return_source_ids=None,
               return_index_map=None):
        """This mesh's self-decomposition into volumetric domains.

        The mesh is read through its own self arrangement — a
        :class:`trueform.CsgGraph` of one operand — whose overlap pockets
        classify into domains; block ``k`` is domain ``ids[k]``, named
        ``str(ids[k])``. See :meth:`trueform.CsgGraph.domains` for what
        each option controls (``selection``, ``exclude_outer_shell``,
        ``ignore_open_fragments``, ``return_source_ids``,
        ``return_index_map``); trueform's defaults apply when omitted. With
        ``return_source_ids=True`` also returns per-cell face provenance as
        two :class:`trueform.OffsetBlockedArray`, parallel to the block
        list, passed through untouched; with ``return_index_map=True``
        returns ``(multiblock, index_map)``, the
        :class:`trueform.DomainsIndexMap` passed through untouched.

        Returns
        -------
        pyvista.MultiBlock
        """
        graph = tf.CsgGraph([self.to_mesh()])
        result = graph.domains(expr, **_forwarded(
            selection=selection, exclude_outer_shell=exclude_outer_shell,
            ignore_open_fragments=ignore_open_fragments,
            return_source_ids=return_source_ids,
            return_index_map=return_index_map))
        if return_index_map:
            cells, ids, index_map = result
            return domains_to_pyvista(cells, ids), index_map
        if return_source_ids:
            cells, ids, tag_blocks, face_blocks = result
            return domains_to_pyvista(cells, ids), tag_blocks, face_blocks
        cells, ids = result
        return domains_to_pyvista(cells, ids)

    def polygon_arrangements(self, *, return_curves=False, mode=None,
                             tolerance=None, resolve_crossings=None,
                             resolve_self_crossings=None,
                             triangulation=None):
        """The mesh split at its own self-intersection curves.

        Per-face provenance rides as ``trueform_face_labels``. See
        :func:`trueform.polygon_arrangements` for what the remaining
        options control (``mode``, ``tolerance``, ``resolve_crossings``,
        ``resolve_self_crossings``, ``triangulation``); trueform's defaults
        apply when omitted.
        """
        result = tf.polygon_arrangements(
            self.to_mesh(), return_curves=return_curves,
            **_forwarded(mode=mode, tolerance=tolerance,
                        resolve_crossings=resolve_crossings,
                        resolve_self_crossings=resolve_self_crossings,
                        triangulation=triangulation))
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

    def cleaned(self, tolerance=None, *, return_index_map=None,
               remove_duplicate_primitives=None,
               remove_unreferenced_points=None):
        """Duplicate vertices and degenerate faces removed.

        ``tolerance`` merges vertices within that world-coordinate
        distance (``None`` keeps exact-duplicate merging only). With
        ``return_index_map=True`` also returns the face and point index
        maps, each a ``(f, kept_ids)`` pair, exactly as
        :func:`trueform.cleaned` returns them for a mesh — passed through
        untouched. See :func:`trueform.cleaned` for what the remaining
        options control (``remove_duplicate_primitives``,
        ``remove_unreferenced_points``); trueform's defaults apply when
        omitted.
        """
        result = tf.cleaned(self.to_mesh(), tolerance, **_forwarded(
            return_index_map=return_index_map,
            remove_duplicate_primitives=remove_duplicate_primitives,
            remove_unreferenced_points=remove_unreferenced_points))
        if return_index_map:
            mesh, face_map, point_map = result
            return to_pyvista(mesh), face_map, point_map
        return to_pyvista(result)

    def triangulated(self):
        """Every face triangulated on its own boundary, shared edges one
        identity in both faces.

        See :func:`trueform.triangulated`.
        """
        return to_pyvista(tf.triangulated(self.to_mesh()))
