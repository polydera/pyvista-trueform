"""
Diagnostics and spatial queries on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import math

import numpy as np
import pyvista as pv
import trueform as tf

from .._conversion import curves_to_pyvista
from . import _operand_mesh


class _QueriesMixin:

    def _query_point(self, point, name):
        point = np.asarray(point, dtype=self.to_mesh().dtype)
        if point.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {point.shape}")
        return point

    def is_closed(self):
        """True when every edge is shared by exactly two faces (watertight)."""
        return tf.is_closed(self.to_mesh())

    def is_manifold(self):
        """True when no edge is shared by more than two faces."""
        return tf.is_manifold(self.to_mesh())

    def area(self):
        """Total surface area. See :func:`trueform.area`."""
        return tf.area(self.to_mesh())

    def volume(self):
        """Enclosed volume of a closed mesh. See :func:`trueform.volume`."""
        return tf.volume(self.to_mesh())

    def ray_cast(self, origin, direction):
        """First mesh face hit by the ray, through the cached spatial tree.

        Returns ``(face_id, t)`` with ``hit = origin + t * direction``, or
        ``None`` when the ray misses. See :func:`trueform.ray_cast`.
        """
        ray = tf.Ray(origin=self._query_point(origin, "origin"),
                     direction=self._query_point(direction, "direction"))
        return tf.ray_cast(ray, self.to_mesh())

    def distance(self, other):
        """Euclidean distance to ``other``.

        ``other`` is a PolyData, a trueform Mesh, or a ``(3,)`` point.
        See :func:`trueform.distance`.
        """
        if isinstance(other, (tf.Mesh, pv.PolyData)):
            return tf.distance(self.to_mesh(), _operand_mesh(other))
        return tf.distance(self.to_mesh(),
                           tf.Point(self._query_point(other, "other")))

    def intersects(self, other):
        """True when this mesh intersects ``other`` (PolyData or trueform
        Mesh). See :func:`trueform.intersects`.
        """
        return tf.intersects(self.to_mesh(), _operand_mesh(other))

    def closest_point(self, query_point):
        """The mesh point closest to ``query_point``.

        Returns ``(face_id, distance_squared, point)`` — the nearest face,
        the squared distance, and the closest point on the mesh. See
        :func:`trueform.neighbor_search`.
        """
        return tf.neighbor_search(
            self.to_mesh(), self._query_point(query_point, "query_point"))

    def closest_point_pair(self, other):
        """The closest witness pair between this mesh and ``other``.

        Returns ``((face_id, other_id), (distance, point, other_point))``
        — the nearest face of this mesh, the witness point on it, the
        witness point on ``other``, and the euclidean distance between
        them (consistent with :meth:`distance`). A faced operand
        (polygonal PolyData or trueform Mesh) answers mesh-to-mesh
        through :func:`trueform.neighbor_search`, and ``other_id`` names
        its nearest face; a faceless one (bare-points PolyData, PointSet)
        queries its points as one batched :class:`trueform.Point`, and
        ``other_id`` names its nearest point.
        """
        mesh = self.to_mesh()
        if isinstance(other, pv.PointSet) or (
                isinstance(other, pv.PolyData)
                and other.GetNumberOfPolys() == 0):
            points = np.asarray(other.points)
            if len(points) == 0:
                raise ValueError("other has no points")
            batch = np.ascontiguousarray(
                points.astype(mesh.dtype, copy=False))
            ids, distances2, witnesses = tf.neighbor_search(
                mesh, tf.Point(batch))
            nearest = int(np.argmin(distances2))
            return ((int(ids[nearest]), nearest),
                    (math.sqrt(float(distances2[nearest])),
                     witnesses[nearest], batch[nearest]))
        pair, (metric, point, other_point) = tf.neighbor_search(
            mesh, _operand_mesh(other))
        return (pair, (math.sqrt(metric), point, other_point))

    def principal_curvatures(self, **kwargs):
        """Per-vertex principal curvatures ``(k0, k1)``; with
        ``directions=True`` also ``(d0, d1)``. See
        :func:`trueform.principal_curvatures` for keyword arguments.
        """
        return tf.principal_curvatures(self.to_mesh(), **kwargs)

    def shape_index(self, **kwargs):
        """Per-vertex shape index in ``[-1, 1]`` (cup to cap). See
        :func:`trueform.shape_index` for keyword arguments.
        """
        return tf.shape_index(self.to_mesh(), **kwargs)

    def boundary_curves(self):
        """The mesh's boundary loops as a line-only PolyData.

        See :func:`trueform.boundary_curves`.
        """
        return curves_to_pyvista(tf.boundary_curves(self.to_mesh()))
