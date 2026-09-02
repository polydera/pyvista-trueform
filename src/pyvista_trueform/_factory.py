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


def csg_graph(datasets, **kwargs):
    """Build a :class:`trueform.CsgGraph` over PyVista datasets.

    One arrangement of N operands, arbitrarily many boolean expressions
    answered against it. Each dataset converts through its own accessor
    cache; a :class:`trueform.Mesh` operand passes through as-is. Operands
    must be triangle meshes (``dataset.triangulate()`` first if needed).

    Whatever the graph answers feeds straight back:
    ``to_pyvista(csg_graph([a, b, c]).mesh(tf.op(0) - tf.op(1)))``.

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
    trueform.CsgGraph
    """
    return tf.CsgGraph(
        [_operand_mesh(dataset) for dataset in datasets], **kwargs)


def _labeled(mesh, labels=None, face_labels=None):
    result = to_pyvista(mesh)
    if labels is not None:
        result.cell_data["trueform_labels"] = labels
    if face_labels is not None:
        result.cell_data["trueform_face_labels"] = face_labels
    return result


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
    prebuilt :class:`trueform.CsgGraph` (use :func:`csg_graph` when the
    build needs non-default arguments, or to answer several reads against
    one arrangement).

    Parameters
    ----------
    datasets_or_graph : sequence of pyvista.PolyData/trueform.Mesh, or trueform.CsgGraph
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
    if not isinstance(graph, tf.CsgGraph):
        graph = csg_graph(datasets_or_graph)
    return domains_to_pyvista(*graph.domains(expr, **kwargs))


__all__ = ["csg_graph", "domains", "mesh_arrangements"]
