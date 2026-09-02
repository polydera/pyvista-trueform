"""
Tube meshes around line PolyData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf
from vtkmodules.util.numpy_support import vtk_to_numpy

from ._conversion import _validated_points, to_pyvista


def _line_paths(dataset):
    if not isinstance(dataset, pv.PolyData):
        raise TypeError(
            f"lines must be a pyvista.PolyData, got {type(dataset).__name__}")
    families = {
        "vertices": dataset.GetNumberOfVerts(),
        "polygons": dataset.GetNumberOfPolys(),
        "strips": dataset.GetNumberOfStrips(),
    }
    present = [name for name, count in families.items() if count]
    if present:
        raise ValueError(
            "PolyData must contain lines only; found " + ", ".join(present))
    if not dataset.GetNumberOfLines():
        raise ValueError("PolyData contains no lines")
    cells = dataset.GetLines()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    if np.all(np.diff(offsets) == 2):
        # Unordered 2-point segments: connect them into polylines first.
        return tf.connect_edges_to_paths(
            np.ascontiguousarray(connectivity.reshape(-1, 2)))
    return tf.OffsetBlockedArray(np.ascontiguousarray(offsets),
                                 np.ascontiguousarray(connectivity))


def tube(lines, radius, n_segments=8):
    """A triangle tube mesh around every polyline, as a fresh PolyData.

    ``lines`` is a line-only PolyData — polylines are swept as they are,
    while 2-point line segments are first connected into polylines through
    :func:`trueform.connect_edges_to_paths` — or a ``(paths, points)``
    pair as trueform's curve producers return. Closed loops are
    auto-detected. See :func:`trueform.make_tube_mesh`.

    Parameters
    ----------
    lines : pyvista.PolyData or (trueform.OffsetBlockedArray, np.ndarray)
        The polylines to sweep.
    radius : float
        Tube radius.
    n_segments : int, default 8
        Vertices per cross-section ring.

    Returns
    -------
    pyvista.PolyData
    """
    if isinstance(lines, tuple) and len(lines) == 2:
        paths, points = lines
        if not isinstance(paths, tf.OffsetBlockedArray):
            raise TypeError(
                "paths must be a trueform.OffsetBlockedArray, "
                f"got {type(paths).__name__}")
    else:
        paths = _line_paths(lines)
        points = np.asarray(lines.points)
    points = np.ascontiguousarray(_validated_points(points))
    return to_pyvista(
        tf.make_tube_mesh((paths, points), radius,
                          radial_segments=n_segments))


__all__ = ["tube"]
