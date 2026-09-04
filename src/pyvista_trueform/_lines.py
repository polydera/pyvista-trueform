"""
Polyline assembly over line PolyData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np

from ._conversion import _line_paths, _validated_points, curves_to_pyvista


def connect_lines(dataset):
    """The dataset's line segments assembled into polylines.

    Unordered 2-point segments connect through
    :func:`trueform.connect_edges_to_paths` — one line cell per polyline,
    a closed loop repeating its first id; cells that already are
    polylines pass through as they are. The result is detached: it keeps
    the dataset's own point ids over a copy of its point array, so later
    edits of the input do not reach it.

    Parameters
    ----------
    dataset : pyvista.PolyData
        Line-only dataset (no vertices, polygons, or strips).

    Returns
    -------
    pyvista.PolyData
    """
    paths = _line_paths(dataset)
    points = np.array(_validated_points(np.asarray(dataset.points)),
                      copy=True, order="C")
    return curves_to_pyvista(paths, points)


__all__ = ["connect_lines"]
