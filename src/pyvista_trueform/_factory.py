"""
N-ary CSG and arrangement entry points over PyVista datasets

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

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
        (``exclude_outer_shell``, ``ignore_open_fragments``,
        ``selection``).

        Returns
        -------
        pyvista.MultiBlock
        """
        return domains_to_pyvista(*self._graph.domains(expr, **kwargs))

    def intersection_curves(self):
        """The arrangement's intersection curves as a line-only PolyData.

        Where two operand surfaces cross (coincident walls excluded). See
        :meth:`trueform.CsgGraph.intersection_curves`.

        Returns
        -------
        pyvista.PolyData
        """
        return curves_to_pyvista(self._graph.intersection_curves())


def csg_graph(datasets, **kwargs):
    """Build a :class:`CsgGraph` over PyVista datasets.

    One arrangement of N operands, arbitrarily many boolean expressions
    answered against it — in PyVista types:
    ``csg_graph([a, b, c]).mesh(tf.op(0) - tf.op(1)).plot()``. Each dataset
    converts through its own accessor cache; a :class:`trueform.Mesh`
    operand passes through as-is. Operands must be triangle meshes
    (``dataset.triangulate()`` first if needed).

    Parameters
    ----------
    datasets : sequence of pyvista.PolyData or trueform.Mesh
        The operands, at least two.
    **kwargs
        Forwarded to :class:`trueform.CsgGraph` (``sheets``, ``mode``,
        ``tolerance``, ``resolve_crossings``, ``within``,
        ``triangulation``).

    Returns
    -------
    CsgGraph
    """
    return CsgGraph(tf.CsgGraph(
        [_operand_mesh(dataset) for dataset in datasets], **kwargs))


def mesh_arrangements(datasets, **kwargs):
    """The arrangement of N PyVista datasets as one labeled PolyData.

    Every face is split along every intersection curve; the source mesh of
    each output face rides as ``trueform_labels``, its source face as
    ``trueform_face_labels``. With ``return_curves=True`` also returns the
    intersection curves as a second, line-only PolyData. See
    :func:`trueform.mesh_arrangements` for keyword arguments.

    Parameters
    ----------
    datasets : sequence of pyvista.PolyData or trueform.Mesh
        The operands, at least two triangle meshes.

    Returns
    -------
    pyvista.PolyData
    """
    meshes = [_operand_mesh(dataset) for dataset in datasets]
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
        The operands, at least two triangle meshes; or the built graph.
    expr : trueform.Expr or int, optional
        Restrict to domains inside the expression's selection.
    **kwargs
        Forwarded to :meth:`trueform.CsgGraph.domains`
        (``exclude_outer_shell``, ``ignore_open_fragments``,
        ``selection``).

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
