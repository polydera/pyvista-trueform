"""
Scalar volumes on the .trueform accessors: the field a mesh states, and the
operations a field answers

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv
import trueform as tf

from .._conversion import (curves_to_pyvista, to_pyvista, volume_to_pyvista,
                           volume_to_trueform)
from .._forward import _forwarded


def _operand_volume(other):
    """The trueform Volume of a second field operand, through its own cache."""
    if isinstance(other, tf.Volume):
        return other
    if isinstance(other, pv.ImageData):
        return other.trueform.to_volume()
    raise TypeError(
        "operand must be a pyvista.ImageData or trueform.Volume, "
        f"got {type(other).__name__}")


class _VolumeMixin:
    """The mesh side: the scalar field a surface states about space."""

    def signed_distance_field(self, dims, spacing, origin, *, dtype=None,
                              mode=None, band=None):
        """This mesh's signed distance field on a stated grid, as ImageData.

        Negative inside the mesh, positive outside, zero on its surface;
        the magnitude is the euclidean distance to the nearest point of
        the surface. The sign is crossing parity, so it wants a CLOSED
        mesh: nothing refuses an open one, it simply has no inside and
        answers a nonnegative field — :meth:`outer_shell` repairs a mesh
        that should have had one. The grid samples this dataset's own
        coordinates: build ``dims``/``spacing``/``origin`` from its
        ``bounds``.

        ``mode="banded"`` measures only the samples within ``band`` voxels
        of the surface and sweeps the far field — an order of magnitude
        faster, with the sign exact everywhere and the magnitude exact
        inside the band. See :func:`trueform.mesh_sdf` for what the
        remaining options control (``dtype``, ``mode``, ``band``);
        trueform's defaults apply when omitted.

        Returns
        -------
        pyvista.ImageData
        """
        return volume_to_pyvista(tf.mesh_sdf(
            self.to_mesh(), dims, spacing, origin,
            **_forwarded(dtype=dtype, mode=mode, band=band)))


class TrueformVolumeAccessor:
    """trueform field operations as ``image.trueform.<method>(...)``.

    The volume sibling of :class:`TrueformAccessor`, registered on
    :class:`pyvista.ImageData` under the same ``.trueform`` name and
    holding the same cache contract: the dataset's active point scalars
    convert into a :class:`trueform.Volume` once, keyed by the dataset's
    VTK modification time, and every call reuses that instance while the
    MTime holds. The conversion copies the samples, so the cache is what
    keeps a 512-cubed scan from being copied per call; when the MTime
    changes the volume is discarded whole and rebuilt.

    .. warning::
        A VTK data array notifies its dataset, so editing samples through
        the dataset (``image.point_data["ct"][k] = ...``,
        ``image.active_scalars[...] = ...``) DOES advance the MTime —
        where the same edit through a PolyData's raw points would not.
        What reaches no VTK object is a write through a NumPy array
        handed to VTK earlier and still held outside, which is exactly
        the buffer :func:`pyvista_trueform.volume_to_pyvista` shares with
        its :class:`trueform.Volume`: after such a write, call
        ``image.Modified()``.

    Which array is the field is PyVista's own question to answer: every
    operation reads the active point scalars, so
    ``image.set_active_scalars("ct")`` selects one, and
    :meth:`to_volume` takes an explicit ``scalars=`` name.

    Examples
    --------
    >>> import pyvista_trueform as tfpv
    >>> field = tfpv.sphere_sdf((32,) * 3, (0.25,) * 3, (-4.0,) * 3,
    ...                         (0.0, 0.0, 0.0), 2.0)
    >>> surface = field.trueform.isosurface()
    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._volume = None
        self._volume_key = None

    def to_volume(self, scalars=None):
        """The cached :class:`trueform.Volume` of this dataset.

        Rebuilt from scratch when the dataset's MTime changes or another
        array is named; otherwise the same instance every call. The key
        is the array, not how it was asked for, so naming the active
        scalars and letting them default are one cache entry. Treat it as
        read-only — edit the PyVista dataset instead.
        """
        named = scalars or self._dataset.point_data.active_scalars_name
        key = (int(self._dataset.GetMTime()), named)
        if self._volume is None or self._volume_key != key:
            self._volume = volume_to_trueform(self._dataset, named)
            self._volume_key = key
        return self._volume

    def isosurface(self, iso=None, *, method=None, refine=None,
                   stabilizer=None, dtype=None):
        """The field's level set as a welded, indexed triangle PolyData.

        A corner is inside when ``sample < iso``, so a signed distance
        field (negative inside) yields outward windings, and a nonzero
        ``iso`` on one is an offset surface. ``method="dual_contouring"``
        places one vertex per surface component of a cell instead of one
        per grid edge, so creases and corners of a distance-like field
        survive; ``"flying_edges"`` (trueform's default) is the fastest
        regular output and is defined for any field. A posed grid answers
        in world space. See :func:`trueform.isosurface` for what the
        remaining options control (``method``, ``refine``, ``stabilizer``,
        ``dtype``); trueform's defaults apply when omitted.

        Returns
        -------
        pyvista.PolyData
        """
        return to_pyvista(tf.isosurface(self.to_volume(), **_forwarded(
            iso=iso, method=method, refine=refine, stabilizer=stabilizer,
            dtype=dtype)))

    def _boolean(self, operation, other, dtype):
        return volume_to_pyvista(tf.volume_boolean(
            self.to_volume(), _operand_volume(other), operation,
            **_forwarded(dtype=dtype)))

    def union(self, other, *, dtype=None):
        """Field union with ``other`` (ImageData or trueform Volume).

        ``min(a, b)`` under the negative-inside convention, so the level
        set of the result is exactly the union of the two solids.
        Matching grids AND matching poses combine sample-wise and keep
        that shared pose; anything else resamples both onto a common
        world-axis-aligned grid. An unsigned field is not a signed
        distance field and trueform refuses it here. See
        :func:`trueform.volume_boolean`.

        Returns
        -------
        pyvista.ImageData
        """
        return self._boolean("union", other, dtype)

    def intersection(self, other, *, dtype=None):
        """Field intersection with ``other`` — ``max(a, b)``.

        The grid, pose and dtype contract of :meth:`union` applies. See
        :func:`trueform.volume_boolean`.

        Returns
        -------
        pyvista.ImageData
        """
        return self._boolean("intersection", other, dtype)

    def difference(self, other, *, dtype=None):
        """Field difference: this field minus ``other`` — ``max(a, -b)``.

        The other direction is the other dataset's accessor:
        ``other.trueform.difference(this)``. The grid, pose and dtype
        contract of :meth:`union` applies. See
        :func:`trueform.volume_boolean`.

        Returns
        -------
        pyvista.ImageData
        """
        return self._boolean("difference", other, dtype)

    def resampled(self, dims, spacing, origin, *, dtype=None):
        """This field regridded onto a stated grid, as fresh ImageData.

        Each target node at ``origin + index * spacing`` takes the
        field's trilinear value there. A grid with no direction regrids
        in its own coordinates, clamped at the edge; a directed one
        regrids THROUGH its direction — the target grid is world space
        and a node outside the posed domain takes a sentinel above the
        field's maximum. The result is axis-aligned: it stands on the
        grid that was asked for. Integer-sampled fields are refused by
        trueform. See :func:`trueform.resampled_volume`.

        Returns
        -------
        pyvista.ImageData
        """
        return volume_to_pyvista(tf.resampled_volume(
            self.to_volume(), dims, spacing, origin,
            **_forwarded(dtype=dtype)))

    def slice_contours(self, plane_origin, u, v, dims2, spacing2, isovalues,
                       *, dtype=None):
        """Isocontours on an oriented slice plane, as 3D line PolyData.

        The field is resampled once onto the plane's 2D grid — node
        ``(i, j)`` at ``plane_origin + i * spacing2[0] * u + j *
        spacing2[1] * v``, stated in this dataset's OWN coordinates, the
        ones its ``origin`` and ``spacing`` speak — and every isovalue is
        contoured on that slice, welded into connected polylines. A
        directed grid answers the same plane, its curves lifted into
        world space by the direction. Contours terminate cleanly at the
        volume boundary. See :func:`trueform.volume_slice_contours`.

        Returns
        -------
        pyvista.PolyData
        """
        return curves_to_pyvista(tf.volume_slice_contours(
            self.to_volume(), plane_origin, u, v, dims2, spacing2, isovalues,
            **_forwarded(dtype=dtype)))
