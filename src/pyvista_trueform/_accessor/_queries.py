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
from .._forward import _forwarded
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

    def is_open(self):
        """True when at least one edge belongs to only one face.
        See :func:`trueform.is_open`.
        """
        return tf.is_open(self.to_mesh())

    def is_manifold(self):
        """True when every vertex's faces are one fan — which also says
        every edge carries at most two of them. Winding does not enter the
        verdict. See :func:`trueform.is_manifold`.
        """
        return tf.is_manifold(self.to_mesh())

    def is_non_manifold(self):
        """True when some edge carries more than two faces, or some
        vertex's faces fall into more than one fan (a bowtie).
        See :func:`trueform.is_non_manifold`.
        """
        return tf.is_non_manifold(self.to_mesh())

    def has_self_intersections(self):
        """True when two non-neighbouring faces of this mesh touch.

        The discovery stops at the first contact, so a mesh that meets
        itself answers without the rest of it being seen. Faces sharing a
        vertex or an edge are neighbours rather than contacts, so a
        bowtie or a fan does not make the verdict true on its own. See
        :func:`trueform.has_self_intersections`.
        """
        return tf.has_self_intersections(self.to_mesh())

    def area(self):
        """Total surface area. See :func:`trueform.area`."""
        return tf.area(self.to_mesh())

    def volume(self):
        """Enclosed volume of a closed mesh. See :func:`trueform.volume`."""
        return tf.volume(self.to_mesh())

    def signed_volume(self):
        """Signed volume of a closed mesh, as a float: positive with
        outward-facing normals, negative with inward-facing ones. See
        :func:`trueform.signed_volume`.
        """
        return tf.signed_volume(self.to_mesh())

    def mean_edge_length(self):
        """Mean edge length over every face's own edges, as a float — a
        shared edge counts once per face holding it. See
        :func:`trueform.mean_edge_length`.
        """
        return tf.mean_edge_length(self.to_mesh())

    def euler_characteristic(self):
        """Euler characteristic ``V - E + F``, each undirected edge counted
        once (boundary and non-manifold edges count like interior ones).
        See :func:`trueform.euler_characteristic`.
        """
        return tf.euler_characteristic(self.to_mesh())

    def face_quality(self):
        """Per-face quality, corner angles, and aspect ratio.

        Returns ``(quality, min_angle, max_angle, aspect_ratio)``, each a
        ``(n_cells,)`` array: the triangle quality measure (1 for
        equilateral, approaching 0 for a sliver, ``-1`` for a face that is
        not a triangle), the smallest and largest corner angle in radians
        (unsigned, in ``[0, pi]``, so a reflex corner reads its
        explement), and the longest side over the shortest. Attach one
        with ``dataset.cell_data["quality"] =
        dataset.trueform.face_quality()[0]``. See
        :func:`trueform.face_quality`.
        """
        return tf.face_quality(self.to_mesh())

    def dihedral_angles(self):
        """The angle each edge two faces share turns through.

        Returns ``(edges, angles)`` — an ``(E, 2)`` array of vertex-id
        pairs naming this dataset's own points, and the aligned ``(E,)``
        angles in radians between the two face normals, so a flat surface
        reads 0. One entry per undirected edge TWO faces share: a
        boundary or non-manifold edge joins no pair and is not stated, so
        ``E`` is smaller than the mesh's edge count and the angles attach
        to the edges, not to this dataset's cells. See
        :func:`trueform.dihedral_angles`.
        """
        return tf.dihedral_angles(self.to_mesh())

    def ray_cast(self, ray, config=None):
        """First mesh face hit by the ray, through the cached spatial tree.

        ``ray`` is a :class:`trueform.Ray`, single or batch; ``config`` is
        the optional ``(min_t, max_t)`` parametric range (scalars, or
        per-ray arrays for a batch), forwarded verbatim. Returns exactly
        what :func:`trueform.ray_cast` returns:

        - single ray: ``(face_id, t)`` with ``hit = origin + t *
          direction``, or ``None`` on a miss;
        - batch ray: ``(face_ids, ts)`` arrays of shape ``(N,)``, a miss
          marked ``-1`` in ``face_ids`` and ``NaN`` in ``ts``.
        """
        if not isinstance(ray, tf.Ray):
            raise TypeError(
                f"ray must be a trueform.Ray, got {type(ray).__name__}")
        return tf.ray_cast(ray, self.to_mesh(), config=config)

    def distance(self, other):
        """Euclidean distance to ``other``.

        ``other`` is a PolyData, a trueform Mesh, a trueform primitive
        (:class:`trueform.Point`, ``Segment``, ``Line``, ``Ray``,
        ``Plane``, ``Triangle``, ``AABB`` — a batched one answers a
        ``(N,)`` array), or a ``(3,)`` point. See
        :func:`trueform.distance`.
        """
        if isinstance(other, (tf.Mesh, pv.PolyData)):
            return tf.distance(self.to_mesh(), _operand_mesh(other))
        if isinstance(other, tf.Primitive):
            return tf.distance(self.to_mesh(), other)
        return tf.distance(self.to_mesh(),
                           tf.Point(self._query_point(other, "other")))

    def signed_distance(self, other):
        """Signed distance from every point of this dataset to ``other``.

        Negative inside ``other``, positive outside; the magnitude is the
        euclidean distance to its surface. ``other`` is a PolyData or
        trueform Mesh with outward-oriented faces. This dataset's points
        go as one batched :class:`trueform.Point`, so a points-only
        dataset queries just as well as a faced one.

        Returns the ``(N,)`` array in the target's dtype — attach it with
        ``dataset.point_data["d"] = dataset.trueform.signed_distance(other)``.
        See :func:`trueform.signed_distance`.
        """
        target = _operand_mesh(other)
        points = np.asarray(self._dataset.points)
        return tf.signed_distance(target, tf.Point(
            np.ascontiguousarray(points.astype(target.dtype, copy=False))))

    def intersects(self, other):
        """True when this mesh intersects ``other`` — a PolyData, a
        trueform Mesh, or a trueform primitive (a batched one answers a
        0/1 ``(N,)`` array). See :func:`trueform.intersects`.
        """
        if isinstance(other, tf.Primitive):
            return tf.intersects(self.to_mesh(), other)
        return tf.intersects(self.to_mesh(), _operand_mesh(other))

    def winding_number(self, query, *, beta=None):
        """This mesh's generalized winding number at ``query``.

        About 1 inside a closed surface, 0 outside, and a graceful
        fractional value for an open sheet or a soup — the query for the
        inputs the exact predicates refuse, and the enclosure count where
        a mesh wraps a point more than once. ``query`` is a ``(3,)``
        point or a :class:`trueform.Point`, single or batched; ``beta``
        is the accuracy knob, larger descending deeper into the tree.
        Returns a float for a single query and a ``(N,)`` array for a
        batched one — always float64, the number being dimensionless
        whatever the mesh's dtype. The winding moments build once and
        cache with the cached mesh's own spatial tree, so repeated
        queries amortize. See :func:`trueform.winding_number`.
        """
        if not isinstance(query, tf.Primitive):
            query = tf.Point(self._query_point(query, "query"))
        return tf.winding_number(self.to_mesh(), query,
                                 **_forwarded(beta=beta))

    def closest_point(self, query_point, *, radius=None):
        """The mesh point closest to ``query_point``.

        ``query_point`` is a ``(3,)`` point or a trueform primitive.
        Returns ``(face_id, distance, point)`` — the nearest face, the
        euclidean distance, and the closest point on the mesh — or
        ``None`` when ``radius`` bounds the search and nothing lies
        within it. A batched primitive answers arrays instead:
        ``(face_ids, distances, points)`` of shapes ``(N,)``, ``(N,)``,
        ``(N, 3)``, a miss marked ``-1`` in ``face_ids``. The underlying
        :func:`trueform.neighbor_search` reports the squared metric; this
        package says "distance" only for euclidean. ``k`` lives on
        :meth:`closest_points`.
        """
        if isinstance(query_point, tf.Primitive):
            query = query_point
        else:
            query = tf.Point(self._query_point(query_point, "query_point"))
        result = tf.neighbor_search(self.to_mesh(), query, radius=radius)
        if result is None:
            return None
        if query.is_batch:
            face_ids, distances2, points = result
            return face_ids, np.sqrt(distances2), points
        face_id, distance2, point = result
        return face_id, math.sqrt(float(distance2)), point

    def closest_points(self, query, k, *, radius=None):
        """The ``k`` nearest mesh hits of ``query``.

        ``query`` is a ``(3,)`` point or a single trueform primitive
        (:class:`trueform.Point`, ``Segment``, ``Triangle``, ``Ray``,
        ...); the answer is a list of up to ``k``
        ``(face_id, distance, point)`` tuples, closest first —
        :meth:`closest_point`'s vocabulary, batched over the neighbors.
        ``radius`` bounds the search, so fewer than ``k`` (or none) may
        answer. A batched primitive answers arrays instead:
        ``(face_ids, distances, points, counts)`` of shapes ``(N, k)``,
        ``(N, k)``, ``(N, k, 3)``, ``(N,)``, with ``-1`` in unfilled id
        slots and ``counts[i]`` the filled ones of query ``i``. The
        underlying :func:`trueform.neighbor_search` reports the squared
        metric; every distance here is euclidean.
        """
        if not isinstance(query, tf.Primitive):
            query = tf.Point(self._query_point(query, "query"))
        result = tf.neighbor_search(self.to_mesh(), query, radius=radius,
                                    k=k)
        if query.is_batch:
            face_ids, distances2, points, counts = result
            return face_ids, np.sqrt(distances2), points, counts
        return [(face_id, math.sqrt(float(distance2)), point)
                for face_id, distance2, point in result]

    def closest_point_pair(self, other, *, radius=None):
        """The closest witness pair between this mesh and ``other``.

        Returns ``((face_id, other_id), (distance, point, other_point))``
        — the nearest face of this mesh, the witness point on it, the
        witness point on ``other``, and the euclidean distance between
        them (consistent with :meth:`distance`) — or ``None`` when
        ``radius`` bounds the search and nothing lies within it. A faced
        operand (polygonal PolyData or trueform Mesh) answers mesh-to-mesh
        through :func:`trueform.neighbor_search`, and ``other_id`` names
        its nearest face; a faceless one (bare-points PolyData, PointSet)
        queries its points as one batched :class:`trueform.Point`, and
        ``other_id`` names its nearest point. ``k`` is excluded, for the
        same reason as :meth:`closest_point`.
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
                mesh, tf.Point(batch), radius=radius)
            nearest = int(np.argmin(distances2))
            if ids[nearest] < 0:
                return None
            return ((int(ids[nearest]), nearest),
                    (math.sqrt(float(distances2[nearest])),
                     witnesses[nearest], batch[nearest]))
        result = tf.neighbor_search(mesh, _operand_mesh(other),
                                    radius=radius)
        if result is None:
            return None
        pair, (metric, point, other_point) = result
        return (pair, (math.sqrt(metric), point, other_point))

    def principal_curvatures(self, *, k=None, directions=None):
        """Per-vertex principal curvatures ``(k0, k1)``; with
        ``directions=True`` also ``(d0, d1)``. ``k`` sets the k-ring
        neighborhood size for the curvature estimate. When omitted,
        trueform's defaults apply — see :func:`trueform.principal_curvatures`.
        """
        return tf.principal_curvatures(
            self.to_mesh(), **_forwarded(k=k, directions=directions))

    def shape_index(self, *, k=None):
        """Per-vertex shape index in ``[-1, 1]`` (cup to cap). ``k`` sets
        the k-ring neighborhood size for the underlying curvature estimate.
        When omitted, trueform's default applies — see
        :func:`trueform.shape_index`.
        """
        return tf.shape_index(self.to_mesh(), **_forwarded(k=k))

    def boundary_curves(self):
        """The mesh's boundary loops as a line-only PolyData.

        See :func:`trueform.boundary_curves`.
        """
        return curves_to_pyvista(tf.boundary_curves(self.to_mesh()))
