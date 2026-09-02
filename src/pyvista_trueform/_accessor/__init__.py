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

from .._conversion import to_pyvista, to_trueform


def _operand_mesh(other):
    """The trueform Mesh of a second operand, through its own cache."""
    if isinstance(other, tf.Mesh):
        return other
    if isinstance(other, pv.PolyData):
        return other.trueform.to_mesh()
    raise TypeError(
        "operand must be a pyvista.PolyData or trueform.Mesh, "
        f"got {type(other).__name__}")


# The mixins import _operand_mesh from this partially initialized module,
# so it must be defined above these imports.
from ._booleans import _BooleansMixin
from ._curves import _CurvesMixin
from ._iso import _IsoMixin
from ._queries import _QueriesMixin
from ._repair import _RepairMixin


class TrueformAccessor(_BooleansMixin, _CurvesMixin, _IsoMixin, _RepairMixin,
                       _QueriesMixin):
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


pv.register_dataset_accessor("trueform", pv.PolyData)(TrueformAccessor)


__all__ = ["TrueformAccessor"]
