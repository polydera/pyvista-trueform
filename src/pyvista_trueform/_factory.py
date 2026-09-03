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

    def mesh(self, expr=None, **kwargs):
        """The boolean result of ``expr`` as a PyVista PolyData.

        With no expression, the full arrangement mesh (every input face,
        cut at intersections). See :meth:`trueform.CsgGraph.mesh` for
        keyword arguments (``selection``, ``inside``,
        ``return_source_ids``, ``return_index_map``). With
        ``return_source_ids=True`` the provenance labels ride as cell data
        (``trueform_labels``, ``trueform_face_labels``); with
        ``return_index_map=True`` returns ``(polydata, index_map)``, the
        :class:`trueform.MeshIndexMap` passed through untouched.

        Returns
        -------
        pyvista.PolyData
        """
        result = self._graph.mesh(expr, **kwargs)
        if kwargs.get("return_index_map"):
            mesh, index_map = result
            return to_pyvista(mesh), index_map
        if kwargs.get("return_source_ids"):
            mesh, labels, face_labels = result
            return _labeled(mesh, labels, face_labels)
        return to_pyvista(result)

    def domains(self, expr=None, **kwargs):
        """Every kept volumetric domain as a block of a PyVista MultiBlock.

        Block ``k`` is domain ``ids[k]``, named ``str(ids[k])``. See
        :meth:`trueform.CsgGraph.domains` for keyword arguments
        (``selection``, ``exclude_outer_shell``, ``ignore_open_fragments``,
        ``return_source_ids``, ``return_index_map``). With
        ``return_source_ids=True`` also returns per-cell face provenance as
        two :class:`trueform.OffsetBlockedArray`, parallel to the block
        list, passed through untouched; with ``return_index_map=True``
        returns ``(multiblock, index_map)``, the
        :class:`trueform.DomainsIndexMap` passed through untouched.

        Returns
        -------
        pyvista.MultiBlock
        """
        result = self._graph.domains(expr, **kwargs)
        if kwargs.get("return_index_map"):
            cells, ids, index_map = result
            return domains_to_pyvista(cells, ids), index_map
        if kwargs.get("return_source_ids"):
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


def csg_graph(datasets, **kwargs):
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
    **kwargs
        Forwarded to :class:`trueform.CsgGraph` (``sheets``, ``mode``,
        ``tolerance``, ``resolve_crossings``, ``within``,
        ``triangulation``).

    Returns
    -------
    CsgGraph
    """
    meshes = _normalized_operands([_operand_mesh(dataset)
                                   for dataset in datasets])
    return CsgGraph(tf.CsgGraph(meshes, **kwargs))


def mesh_arrangements(datasets, **kwargs):
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
    **kwargs
        Forwarded to :func:`trueform.mesh_arrangements` (``mode``,
        ``tolerance``, ``resolve_crossings``, ``resolve_self_crossings``,
        ``within``, ``triangulation``, in addition to ``return_curves``
        above).

    Returns
    -------
    pyvista.PolyData
    """
    meshes = _normalized_operands([_operand_mesh(dataset)
                                   for dataset in datasets])
    result = tf.mesh_arrangements(meshes, **kwargs)
    if kwargs.get("return_curves"):
        mesh, tag_labels, face_labels, curves = result
        return (_labeled(mesh, tag_labels, face_labels),
                curves_to_pyvista(curves))
    mesh, tag_labels, face_labels = result
    return _labeled(mesh, tag_labels, face_labels)


def domains(datasets_or_graph, expr=None, **kwargs):
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
    **kwargs
        Forwarded to :meth:`CsgGraph.domains`
        (``selection``, ``exclude_outer_shell``, ``ignore_open_fragments``,
        ``return_source_ids``, ``return_index_map``).

    Returns
    -------
    pyvista.MultiBlock
    """
    graph = datasets_or_graph
    if isinstance(graph, tf.CsgGraph):
        graph = CsgGraph(graph)
    elif not isinstance(graph, CsgGraph):
        graph = csg_graph(datasets_or_graph)
    return graph.domains(expr, **kwargs)


__all__ = ["CsgGraph", "csg_graph", "domains", "mesh_arrangements"]
