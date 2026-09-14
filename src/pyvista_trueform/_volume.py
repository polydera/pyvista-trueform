"""
Analytic scalar fields as fresh PyVista ImageData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from ._conversion import volume_to_pyvista
from ._forward import _forwarded


def sphere_sdf(dims, spacing, origin, center, radius, *, dtype=None):
    """The signed distance field of a sphere on a regular grid.

    Each sample holds ``distance(point, center) - radius``: negative
    inside, zero on the surface, positive outside. The field rides as the
    ``trueform_samples`` point-data array of an axis-aligned ImageData,
    so it is one ``.trueform.isosurface()`` away from a mesh and one
    ``.trueform.union(other)`` away from a bigger solid. A mesh's own
    field comes from ``dataset.trueform.signed_distance_field(...)``.
    See :func:`trueform.sphere_sdf`.

    Parameters
    ----------
    dims : sequence of 3 ints
        Number of samples along x, y, z.
    spacing : sequence of 3 floats
        Physical size of one voxel step along x, y, z.
    origin : sequence of 3 floats
        Position of sample ``(0, 0, 0)``.
    center : sequence of 3 floats
        Sphere center, in the grid's own space.
    radius : float
        Sphere radius.
    dtype : numpy.dtype, optional
        Sample dtype, float32 or float64. Trueform's default (float32)
        applies when omitted.

    Returns
    -------
    pyvista.ImageData
    """
    return volume_to_pyvista(tf.sphere_sdf(
        dims, spacing, origin, center, radius, **_forwarded(dtype=dtype)))


__all__ = ["sphere_sdf"]
