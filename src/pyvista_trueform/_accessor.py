"""
The .trueform accessor on pyvista.PolyData

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf

from ._conversion import curves_to_pyvista, to_pyvista, to_trueform


def _operand_mesh(other):
    """The trueform Mesh of a second operand, through its own cache."""
    if isinstance(other, tf.Mesh):
        return other
    if isinstance(other, pv.PolyData):
        accessor = getattr(other, "trueform", None)
        if accessor is not None:
            return accessor.to_mesh()
        return to_trueform(other)
    raise TypeError(
        "operand must be a pyvista.PolyData or trueform.Mesh, "
        f"got {type(other).__name__}")


class TrueformAccessor:
    """trueform operations exposed as ``polydata.trueform.<method>(...)``.

    The accessor converts the dataset into a :class:`trueform.Mesh` once and
    caches that ENTIRE mesh keyed by the dataset's VTK modification time —
    one integer compare per access. While the MTime is unchanged every call
    reuses the same Mesh instance, so its lazily built structures (spatial
    tree, face membership, manifold edge link) amortize across calls. When
    the MTime changes the cached mesh is discarded whole and a fresh one is
    built — whole-value replacement, never partial refresh.

    .. warning::
        VTK only advances the MTime through its own API. Mutating a raw
        NumPy view of the underlying arrays (e.g. ``np.asarray(pd.points)``)
        does not bump it, so the accessor would keep serving the stale mesh.
        Call ``polydata.Modified()`` after such edits. Assignments through
        PyVista's own surface (``pd.points = ...``, ``pd.points[0] = ...``)
        notify VTK already.

    Every operation that returns geometry returns a fresh
    :class:`pyvista.PolyData`; trueform's label arrays ride along verbatim
    as cell data (``trueform_labels``, ``trueform_face_labels``). Everything
    else in trueform stays reachable by composition: ``to_mesh()`` into the
    trueform API, :func:`pyvista_trueform.to_pyvista` back.

    Examples
    --------
    >>> import pyvista as pv
    >>> import pyvista_trueform  # registers the accessor
    >>> a = pv.Cube().triangulate()
    >>> b = pv.Cube(center=(0.5, 0.5, 0.5)).triangulate()
    >>> result = a.trueform.union(b)
    """

    def __init__(self, dataset):
        self._dataset = dataset
        self._mesh = None
        self._mesh_mtime = None

    # -- the cache -------------------------------------------------------

    def to_mesh(self):
        """The cached :class:`trueform.Mesh` of this dataset.

        Rebuilt from scratch when the dataset's MTime changes; otherwise the
        same instance every call, so trueform's lazy structures amortize.
        Treat it as read-only — edit the PyVista dataset instead.
        """
        mtime = int(self._dataset.GetMTime())
        if self._mesh is None or self._mesh_mtime != mtime:
            self._mesh = to_trueform(self._dataset)
            self._mesh_mtime = mtime
        return self._mesh

    def _scalar_field(self, scalars):
        """A point-data array by name, or any array-like, in the mesh dtype."""
        if isinstance(scalars, str):
            scalars = self._dataset.point_data[scalars]
        scalars = np.asarray(scalars)
        return np.ascontiguousarray(
            scalars.astype(self.to_mesh().dtype, copy=False))

    def _labeled(self, mesh, labels=None, face_labels=None):
        result = to_pyvista(mesh)
        if labels is not None:
            result.cell_data["trueform_labels"] = labels
        if face_labels is not None:
            result.cell_data["trueform_face_labels"] = face_labels
        return result

    def _boolean(self, operation, other, return_curves):
        result = operation(self.to_mesh(), _operand_mesh(other),
                           return_curves=return_curves)
        if return_curves:
            mesh, labels, face_labels, curves = result
            return (self._labeled(mesh, labels, face_labels),
                    curves_to_pyvista(curves))
        mesh, labels, face_labels = result
        return self._labeled(mesh, labels, face_labels)

    # -- booleans --------------------------------------------------------

    def union(self, other, *, return_curves=False):
        """Boolean union with ``other`` (PolyData or trueform Mesh).

        With ``return_curves=True`` also returns the intersection curves as
        a second, line-only PolyData. See :func:`trueform.boolean_union`.
        """
        return self._boolean(tf.boolean_union, other, return_curves)

    def intersection(self, other, *, return_curves=False):
        """Boolean intersection with ``other`` (PolyData or trueform Mesh).

        See :func:`trueform.boolean_intersection`.
        """
        return self._boolean(tf.boolean_intersection, other, return_curves)

    def difference(self, other, *, return_curves=False):
        """Boolean difference: this mesh minus ``other``.

        The other direction is the other dataset's accessor:
        ``other.trueform.difference(this)``. See
        :func:`trueform.boolean_difference`.
        """
        return self._boolean(tf.boolean_difference, other, return_curves)

    # -- curves ----------------------------------------------------------

    def intersection_curves(self, other, **kwargs):
        """Intersection curves with ``other`` as a line-only PolyData.

        See :func:`trueform.intersection_curves` for keyword arguments.
        """
        return curves_to_pyvista(
            tf.intersection_curves(self.to_mesh(), _operand_mesh(other),
                                   **kwargs))

    def self_intersection_curves(self, **kwargs):
        """This mesh's self-intersection curves as a line-only PolyData.

        See :func:`trueform.self_intersection_curves` for keyword arguments.
        """
        return curves_to_pyvista(
            tf.self_intersection_curves(self.to_mesh(), **kwargs))

    # -- self-intersection repair ----------------------------------------

    def polygon_arrangements(self, *, return_curves=False, **kwargs):
        """The mesh split at its own self-intersection curves.

        Per-face provenance rides as ``trueform_face_labels``. See
        :func:`trueform.polygon_arrangements` for keyword arguments.
        """
        result = tf.polygon_arrangements(
            self.to_mesh(), return_curves=return_curves, **kwargs)
        if return_curves:
            mesh, face_labels, curves = result
            return (self._labeled(mesh, face_labels=face_labels),
                    curves_to_pyvista(curves))
        mesh, face_labels = result
        return self._labeled(mesh, face_labels=face_labels)

    def outer_shell(self):
        """Repair to the outer shell: the boundary of the union of
        everything the mesh encloses, free of self-intersections.

        See :func:`trueform.outer_shell`.
        """
        return to_pyvista(tf.outer_shell(self.to_mesh()))

    # -- scalar fields ---------------------------------------------------

    def isocontours(self, scalars, threshold):
        """Isocontour curves of a per-vertex scalar field as line PolyData.

        ``scalars`` is a ``point_data`` array name or an array;
        ``threshold`` a value or an array of values. See
        :func:`trueform.isocontours`.
        """
        return curves_to_pyvista(
            tf.isocontours(self.to_mesh(), self._scalar_field(scalars),
                           threshold))

    def isobands(self, scalars, cut_values, *, selected_bands=None,
                 return_curves=False):
        """The mesh recut into scalar-field bands.

        The band of every face rides as ``trueform_labels``, its source face
        as ``trueform_face_labels``. See :func:`trueform.isobands`.
        """
        result = tf.isobands(self.to_mesh(), self._scalar_field(scalars),
                             cut_values, selected_bands=selected_bands,
                             return_curves=return_curves)
        if return_curves:
            mesh, labels, face_labels, curves = result
            return (self._labeled(mesh, labels, face_labels),
                    curves_to_pyvista(curves))
        mesh, labels, face_labels = result
        return self._labeled(mesh, labels, face_labels)

    # -- processing ------------------------------------------------------

    def cleaned(self, tolerance=None, **kwargs):
        """Duplicate vertices and degenerate faces removed.

        See :func:`trueform.cleaned` for keyword arguments.
        """
        return to_pyvista(tf.cleaned(self.to_mesh(), tolerance, **kwargs))

    def triangulated(self):
        """Every face triangulated on its own boundary, shared edges one
        identity in both faces.

        See :func:`trueform.triangulated`.
        """
        return to_pyvista(tf.triangulated(self.to_mesh()))

    # -- diagnostics -----------------------------------------------------

    def is_closed(self):
        """True when every edge is shared by exactly two faces (watertight)."""
        return tf.is_closed(self.to_mesh())

    def is_manifold(self):
        """True when no edge is shared by more than two faces."""
        return tf.is_manifold(self.to_mesh())


if hasattr(pv, "register_dataset_accessor"):
    pv.register_dataset_accessor("trueform", pv.PolyData)(TrueformAccessor)


__all__ = ["TrueformAccessor"]
