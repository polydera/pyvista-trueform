"""
N-ary CSG and arrangement entry points over PyVista datasets

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import trueform as tf

from ._accessor import _operand_mesh
from ._conversion import curves_to_pyvista, domains_to_pyvista, to_pyvista
from ._forward import _forwarded


def _labeled(mesh, labels=None, face_labels=None):
    result = to_pyvista(mesh)
    if labels is not None:
        result.cell_data["trueform_labels"] = labels
    if face_labels is not None:
        result.cell_data["trueform_face_labels"] = face_labels
    return result


def _carrying_transformation(mesh, source):
    if source.transformation is not None:
        mesh.transformation = source.transformation
    return mesh


def _as_dynamic(mesh):
    """The same mesh with triangle faces re-expressed as dynamic blocks."""
    return _carrying_transformation(
        tf.Mesh(tf.as_offset_blocked(mesh.faces), mesh.points), mesh)


def _widened(mesh):
    """The same mesh with its faces widened to int64."""
    if mesh.is_dynamic:
        faces = tf.OffsetBlockedArray(mesh.faces.offsets.astype(np.int64),
                                      mesh.faces.data.astype(np.int64))
    else:
        faces = mesh.faces.astype(np.int64)
    return _carrying_transformation(tf.Mesh(faces, mesh.points), mesh)


def _normalized_operands(meshes):
    """Meshes ready for one arrangement build: one representation and one
    index dtype, generalizing trueform's own pairwise boolean
    ``_normalized_pair`` (``python/src/trueform/_csg/boolean.py``) to N
    operands. All-triangle stays triangle; anything else re-expresses every
    operand as dynamic blocks, losslessly. A differing index dtype widens
    every operand to int64.
    """
    if not all(mesh.ngon == 3 and not mesh.is_dynamic for mesh in meshes):
        meshes = [mesh if mesh.is_dynamic else _as_dynamic(mesh)
                 for mesh in meshes]
    wide = np.dtype(np.int64)
    if len({mesh.faces.dtype for mesh in meshes}) > 1:
        meshes = [mesh if mesh.faces.dtype == wide else _widened(mesh)
                 for mesh in meshes]
    return meshes


class CsgGraph:
    """A :class:`trueform.CsgGraph` whose readers answer in PyVista types.

    Built by :func:`csg_graph`. The wrapper holds the native graph and
    nothing else; :meth:`mesh`, :meth:`domains`, and
    :meth:`intersection_curves` forward to it and convert the results, so
    the graph reads straight into PyVista:
    ``csg_graph([a, b, c]).mesh(tf.op(0) - tf.op(1)).plot()``. Everything
    the wrapper does not convert lives on :attr:`native`.
    """

    def __init__(self, graph):
        self._graph = graph

    @property
    def native(self):
        """The underlying :class:`trueform.CsgGraph`.

        The escape hatch: everything the wrapper does not convert —
        ``created_points``, ``forms``, ``sheets``, and the rest of the
        construction state — lives here, in trueform's own types.
        """
        return self._graph

    def mesh(self, expr=None, *, selection=None, inside=None,
             return_source_ids=None, return_index_map=None):
        """The boolean result of ``expr`` as a PyVista PolyData.

        With no expression, the full arrangement mesh (every input face,
        cut at intersections). See :meth:`trueform.CsgGraph.mesh` for what
        each option controls (``selection``, ``inside``,
        ``return_source_ids``, ``return_index_map``); trueform's defaults
        apply when omitted. With ``return_source_ids=True`` the provenance
        labels ride as cell data (``trueform_labels``,
        ``trueform_face_labels``); with ``return_index_map=True`` returns
        ``(polydata, index_map)``, the :class:`trueform.MeshIndexMap`
        passed through untouched.

        Returns
        -------
        pyvista.PolyData
        """
        result = self._graph.mesh(expr, **_forwarded(
            selection=selection, inside=inside,
            return_source_ids=return_source_ids,
            return_index_map=return_index_map))
        if return_index_map:
            mesh, index_map = result
            return to_pyvista(mesh), index_map
        if return_source_ids:
            mesh, labels, face_labels = result
            return _labeled(mesh, labels, face_labels)
        return to_pyvista(result)

    def domains(self, expr=None, *, selection=None, exclude_outer_shell=None,
               ignore_open_fragments=None, return_source_ids=None,
               return_index_map=None):
        """Every kept volumetric domain as a block of a PyVista MultiBlock.

        Block ``k`` is domain ``ids[k]``, named ``str(ids[k])``. See
        :meth:`trueform.CsgGraph.domains` for what each option controls
        (``selection``, ``exclude_outer_shell``, ``ignore_open_fragments``,
        ``return_source_ids``, ``return_index_map``); trueform's defaults
        apply when omitted. With ``return_source_ids=True`` also returns
        per-cell face provenance as two :class:`trueform.OffsetBlockedArray`,
        parallel to the block list, passed through untouched; with
        ``return_index_map=True`` returns ``(multiblock, index_map)``, the
        :class:`trueform.DomainsIndexMap` passed through untouched.

        Returns
        -------
        pyvista.MultiBlock
        """
        result = self._graph.domains(expr, **_forwarded(
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

    def intersection_curves(self):
        """The arrangement's intersection curves as a line-only PolyData.

        Where two operand surfaces cross (coincident walls excluded). See
        :meth:`trueform.CsgGraph.intersection_curves`.

        Returns
        -------
        pyvista.PolyData
        """
        return curves_to_pyvista(self._graph.intersection_curves())

    def outer_shell(self):
        """The arrangement's outer shell as a PyVista PolyData.

        The boundary between the unbounded universe and everything the
        operands enclose, oriented outward — a structural read off the
        graph already built, no second arrangement. See
        :meth:`trueform.CsgGraph.outer_shell`.

        Returns
        -------
        pyvista.PolyData
        """
        return to_pyvista(self._graph.outer_shell())


def csg_graph(datasets, *, sheets=None, mode=None, tolerance=None,
             resolve_crossings=None, within=None, triangulation=None):
    """Build a :class:`CsgGraph` over PyVista datasets.

    One arrangement of N operands, arbitrarily many boolean expressions
    answered against it — in PyVista types:
    ``csg_graph([a, b, c]).mesh(tf.op(0) - tf.op(1)).plot()``. Each dataset
    converts through its own accessor cache; a :class:`trueform.Mesh`
    operand passes through as-is. Operands stay a triangle graph when every
    one is all-triangle; otherwise every operand is re-expressed as dynamic
    (variable-sized) faces first — lossless, mirroring how trueform's own
    booleans normalize a mixed pair. A single operand is legal: its own
    self arrangement.

    Parameters
    ----------
    datasets : sequence of pyvista.PolyData or trueform.Mesh
        The operands.
    sheets, mode, tolerance, resolve_crossings, within, triangulation
        Forwarded to :class:`trueform.CsgGraph`; trueform's defaults apply
        when omitted. See :class:`trueform.CsgGraph` for what each controls.

    Returns
    -------
    CsgGraph
    """
    meshes = _normalized_operands([_operand_mesh(dataset)
                                   for dataset in datasets])
    return CsgGraph(tf.CsgGraph(meshes, **_forwarded(
        sheets=sheets, mode=mode, tolerance=tolerance,
        resolve_crossings=resolve_crossings, within=within,
        triangulation=triangulation)))


def mesh_arrangements(datasets, *, return_curves=False, mode=None,
                      tolerance=None, resolve_crossings=None,
                      resolve_self_crossings=None, within=None,
                      triangulation=None):
    """The arrangement of N PyVista datasets as one labeled PolyData.

    Every face is split along every intersection curve; the source mesh of
    each output face rides as ``trueform_labels``, its source face as
    ``trueform_face_labels``. With ``return_curves=True`` also returns the
    intersection curves as a second, line-only PolyData. Operands stay a
    triangle mesh when every one is all-triangle; otherwise every operand
    is re-expressed as dynamic faces first, as in :func:`csg_graph`.

    Parameters
    ----------
    datasets : sequence of pyvista.PolyData or trueform.Mesh
        The operands, two or more.
    return_curves, mode, tolerance, resolve_crossings, resolve_self_crossings, within, triangulation
        Forwarded to :func:`trueform.mesh_arrangements`; trueform's
        defaults apply when omitted (except ``return_curves``, whose
        default lives here since this wrapper reads it to shape its own
        return value).

    Returns
    -------
    pyvista.PolyData
    """
    meshes = _normalized_operands([_operand_mesh(dataset)
                                   for dataset in datasets])
    result = tf.mesh_arrangements(meshes, return_curves=return_curves,
                                  **_forwarded(
                                      mode=mode, tolerance=tolerance,
                                      resolve_crossings=resolve_crossings,
                                      resolve_self_crossings=resolve_self_crossings,
                                      within=within,
                                      triangulation=triangulation))
    if return_curves:
        mesh, tag_labels, face_labels, curves = result
        return (_labeled(mesh, tag_labels, face_labels),
                curves_to_pyvista(curves))
    mesh, tag_labels, face_labels = result
    return _labeled(mesh, tag_labels, face_labels)


def domains(datasets_or_graph, expr=None, *, selection=None,
           exclude_outer_shell=None, ignore_open_fragments=None,
           return_source_ids=None, return_index_map=None):
    """Every kept volumetric domain as a block of a PyVista MultiBlock.

    The arrangement of the operands partitions space into watertight
    domains; block ``k`` is domain ``ids[k]``, named ``str(ids[k])``.
    Accepts either a sequence of datasets — the graph is built here — or a
    prebuilt graph, :class:`CsgGraph` or raw :class:`trueform.CsgGraph`
    (use :func:`csg_graph` when the build needs non-default arguments, or
    to answer several reads against one arrangement).

    Parameters
    ----------
    datasets_or_graph : sequence of pyvista.PolyData/trueform.Mesh, or CsgGraph, or trueform.CsgGraph
        The operands, or the built graph.
    expr : trueform.Expr or int, optional
        Restrict to domains inside the expression's selection.
    selection, exclude_outer_shell, ignore_open_fragments, return_source_ids, return_index_map
        Forwarded to :meth:`CsgGraph.domains`; trueform's defaults apply
        when omitted.

    Returns
    -------
    pyvista.MultiBlock
    """
    graph = datasets_or_graph
    if isinstance(graph, tf.CsgGraph):
        graph = CsgGraph(graph)
    elif not isinstance(graph, CsgGraph):
        graph = csg_graph(datasets_or_graph)
    return graph.domains(expr, selection=selection,
                         exclude_outer_shell=exclude_outer_shell,
                         ignore_open_fragments=ignore_open_fragments,
                         return_source_ids=return_source_ids,
                         return_index_map=return_index_map)


def split_into_domains(arranged, *, ignore_open_fragments=None,
                       exclude_outer_shell=None):
    """Split an already-arranged mesh into its volumetric domains.

    Reads a dataset whose faces already stop at every intersection — a
    :func:`mesh_arrangements` output, a :meth:`CsgGraph.mesh` read — with
    :func:`trueform.domain_labels` and splits it with
    :func:`trueform.split_into_domains`: no arrangement is built, the
    labeling walks the faces as they are. :meth:`TrueformAccessor.domains`
    is the other entry: it arranges the dataset from scratch through its
    own self :class:`trueform.CsgGraph`, so it is the one to call on a
    mesh that still self-intersects. Block ``k`` is domain ``ids[k]``,
    named ``str(ids[k])``, watertight with outward-of-domain normals.

    Parameters
    ----------
    arranged : pyvista.PolyData or trueform.Mesh
        The already-cut mesh; a PolyData converts through its own
        accessor cache.
    ignore_open_fragments, exclude_outer_shell
        Forwarded to :func:`trueform.domain_labels`; trueform's defaults
        apply when omitted — note that by default the unbounded universe
        is a domain too, where the graph readers exclude it.

    Returns
    -------
    pyvista.MultiBlock
    """
    mesh = _operand_mesh(arranged)
    labels = tf.domain_labels(mesh, **_forwarded(
        ignore_open_fragments=ignore_open_fragments,
        exclude_outer_shell=exclude_outer_shell))
    cells, ids = tf.split_into_domains(mesh, labels)
    return domains_to_pyvista(cells, ids)


__all__ = ["CsgGraph", "csg_graph", "domains", "mesh_arrangements",
           "split_into_domains"]
