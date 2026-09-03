"""
Mesh primitive generators as fresh PyVista PolyData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from ._conversion import to_pyvista
from ._forward import _forwarded


def box(width, height, depth, *, width_ticks=None, height_ticks=None,
        depth_ticks=None, dtype=None, index_dtype=None):
    """An axis-aligned box mesh centered at the origin.

    Outward-facing normals (CCW winding). With every tick at trueform's
    default of 1, the box is 8 points and 12 triangles; higher ticks
    subdivide each axis with shared vertices.
    See :func:`trueform.make_box_mesh`.

    Parameters
    ----------
    width, height, depth : float
        Extent along x, y, z.
    width_ticks, height_ticks, depth_ticks : int, optional
        Subdivisions along x, y, z. Trueform's default (1, no subdivision)
        applies when omitted.
    dtype, index_dtype : numpy.dtype, optional
        Point and face-index dtype. Trueform's defaults (float32, int32)
        apply when omitted.

    Returns
    -------
    pyvista.PolyData
    """
    return to_pyvista(tf.make_box_mesh(width, height, depth, **_forwarded(
        width_ticks=width_ticks, height_ticks=height_ticks,
        depth_ticks=depth_ticks, dtype=dtype, index_dtype=index_dtype)))


def sphere(radius, *, stacks=None, segments=None, dtype=None,
           index_dtype=None):
    """A UV sphere mesh centered at the origin.

    Latitude/longitude subdivision, outward-facing normals (CCW winding).
    See :func:`trueform.make_sphere_mesh`.

    Parameters
    ----------
    radius : float
        Sphere radius.
    stacks, segments : int, optional
        Latitude and longitude subdivisions. Trueform's default (20, 20)
        applies when omitted.
    dtype, index_dtype : numpy.dtype, optional
        Point and face-index dtype. Trueform's defaults (float32, int32)
        apply when omitted.

    Returns
    -------
    pyvista.PolyData
    """
    return to_pyvista(tf.make_sphere_mesh(radius, **_forwarded(
        stacks=stacks, segments=segments, dtype=dtype,
        index_dtype=index_dtype)))


def cylinder(radius, height, *, segments=None, dtype=None,
             index_dtype=None):
    """A capped cylinder mesh centered at the origin along the z-axis.

    Extends from z = -height/2 to z = +height/2, outward-facing normals
    (CCW winding). See :func:`trueform.make_cylinder_mesh`.

    Parameters
    ----------
    radius, height : float
        Cylinder radius and height.
    segments : int, optional
        Subdivisions around the circumference. Trueform's default (20)
        applies when omitted.
    dtype, index_dtype : numpy.dtype, optional
        Point and face-index dtype. Trueform's defaults (float32, int32)
        apply when omitted.

    Returns
    -------
    pyvista.PolyData
    """
    return to_pyvista(tf.make_cylinder_mesh(radius, height, **_forwarded(
        segments=segments, dtype=dtype, index_dtype=index_dtype)))


def plane(width, height, *, width_ticks=None, height_ticks=None,
          dtype=None, index_dtype=None):
    """A flat rectangular plane mesh in the XY plane, centered at the origin.

    Normal points +z (CCW winding). With every tick at trueform's default
    of 1, the plane is 4 points and 2 triangles; higher ticks subdivide
    each axis with shared vertices. See :func:`trueform.make_plane_mesh`.

    Parameters
    ----------
    width, height : float
        Extent along x, y.
    width_ticks, height_ticks : int, optional
        Subdivisions along x, y. Trueform's default (1, no subdivision)
        applies when omitted.
    dtype, index_dtype : numpy.dtype, optional
        Point and face-index dtype. Trueform's defaults (float32, int32)
        apply when omitted.

    Returns
    -------
    pyvista.PolyData
    """
    return to_pyvista(tf.make_plane_mesh(width, height, **_forwarded(
        width_ticks=width_ticks, height_ticks=height_ticks, dtype=dtype,
        index_dtype=index_dtype)))


__all__ = ["box", "cylinder", "plane", "sphere"]
