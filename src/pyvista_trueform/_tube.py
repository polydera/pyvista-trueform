"""
Tube meshes around line PolyData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import trueform as tf

from ._conversion import _line_paths, _validated_points, to_pyvista
from ._forward import _forwarded


def tube(lines, radius, *, n_segments=None):
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
    n_segments : int, optional
        Vertices per cross-section ring (``radial_segments`` in
        :func:`trueform.make_tube_mesh`). Trueform's default applies when
        omitted.

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
                          **_forwarded(radial_segments=n_segments)))


__all__ = ["tube"]
