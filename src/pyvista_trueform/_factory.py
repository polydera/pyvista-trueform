"""
N-ary CSG entry point over PyVista datasets

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from ._accessor import _operand_mesh


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


__all__ = ["csg_graph"]
