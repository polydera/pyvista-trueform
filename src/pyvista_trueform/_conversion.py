"""
Conversions between trueform and PyVista geometry

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkCellArray

_POINT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))


def _validated_points(points, name="points"):
    if not isinstance(points, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray, got {type(points).__name__}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if points.dtype not in _POINT_DTYPES:
        raise TypeError(
            f"{name} must have dtype float32 or float64, got {points.dtype}")
    return points


def _validated_polydata(dataset):
    if not isinstance(dataset, pv.PolyData):
        raise TypeError(
            f"dataset must be a pyvista.PolyData, got {type(dataset).__name__}")
    families = {
        "vertices": dataset.GetNumberOfVerts(),
        "lines": dataset.GetNumberOfLines(),
        "strips": dataset.GetNumberOfStrips(),
    }
    present = [name for name, count in families.items() if count]
    if present:
        raise ValueError(
            "PolyData must contain polygons only; found " + ", ".join(present))
    return dataset


def _mesh_arrays(geometry):
    if isinstance(geometry, tf.Mesh):
        return geometry.faces, geometry.points, geometry.transformation
    if isinstance(geometry, tuple) and len(geometry) == 2:
        faces, points = geometry
        return faces, points, None
    raise TypeError(
        "geometry must be a trueform.Mesh or a (faces, points) tuple, "
        f"got {type(geometry).__name__}")


def _transformed_points(points, transformation):
    transformed = points @ transformation[:3, :3].T + transformation[:3, 3]
    return np.ascontiguousarray(transformed.astype(points.dtype, copy=False))


def _cell_array(offsets, connectivity):
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtk(np.ascontiguousarray(offsets), deep=False),
        numpy_to_vtk(np.ascontiguousarray(connectivity), deep=False),
    )
    return cells


def to_trueform(dataset):
    """Copy polygonal PyVista geometry into a fresh trueform Mesh.

    Faces convert straight off VTK 9's cell-array layout — the offsets and
    connectivity arrays ARE trueform's offset-block model. An all-triangle
    dataset yields fixed ``(N, 3)`` faces, anything else a
    :class:`trueform.OffsetBlockedArray`; face indices keep the dataset's VTK
    storage width (int32 or int64).

    The mesh is detached: later PyVista edits do not affect it. Retain it
    when several trueform operations should share its lazily built
    structures (tree, face membership, edge link) — or use the
    ``dataset.trueform`` accessor, which caches exactly this conversion.

    Parameters
    ----------
    dataset : pyvista.PolyData
        Polygon-only dataset (no vertices, lines, or strips).

    Returns
    -------
    trueform.Mesh
    """
    _validated_polydata(dataset)
    points = np.array(
        _validated_points(np.asarray(dataset.points)), copy=True, order="C")
    polygons = dataset.GetPolys()
    connectivity = np.array(
        vtk_to_numpy(polygons.GetConnectivityArray()), copy=True, order="C")
    if dataset.is_all_triangles:
        faces = connectivity.reshape(-1, 3)
    else:
        offsets = np.array(
            vtk_to_numpy(polygons.GetOffsetsArray()), copy=True, order="C")
        faces = tf.OffsetBlockedArray(offsets, connectivity)
    return tf.Mesh(faces, points)


def to_pyvista(geometry, *, apply_transformation=True):
    """Convert trueform polygon geometry to a fresh PyVista PolyData.

    Fixed ``(N, 3)`` faces go through
    :meth:`pyvista.PolyData.from_regular_faces` with ``deep=False``; dynamic
    faces go through ``vtkCellArray.SetData(offsets, connectivity)`` —
    trueform's offset-block faces ARE VTK 9's cell-array layout. Both paths
    are zero-copy: VTK retains the NumPy buffers, so the result stays valid
    after the inputs are released.

    Parameters
    ----------
    geometry : trueform.Mesh or (faces, points) tuple
        ``faces`` is an ``(N, 3)`` int32/int64 array or a
        :class:`trueform.OffsetBlockedArray`; ``points`` is ``(P, 3)``
        float32/float64.
    apply_transformation : bool, default True
        Bake a :class:`trueform.Mesh` transformation into the exported
        points (that path copies the points, by necessity).

    Returns
    -------
    pyvista.PolyData
    """
    faces, points, transformation = _mesh_arrays(geometry)
    points = np.ascontiguousarray(_validated_points(points))
    if apply_transformation and transformation is not None:
        points = _transformed_points(points, transformation)

    if isinstance(faces, tf.OffsetBlockedArray):
        result = pv.PolyData()
        result.SetPoints(pv.vtk_points(points, deep=False))
        result.SetPolys(_cell_array(faces.offsets, faces.data))
        return result
    if isinstance(faces, np.ndarray):
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"fixed faces must have shape (N, 3), got {faces.shape}")
        if faces.dtype not in _INDEX_DTYPES:
            raise TypeError(
                f"faces must have dtype int32 or int64, got {faces.dtype}")
        return pv.PolyData.from_regular_faces(
            points, np.ascontiguousarray(faces), deep=False)
    raise TypeError(
        "faces must be a numpy.ndarray or trueform.OffsetBlockedArray, "
        f"got {type(faces).__name__}")


def curves_to_pyvista(paths, points=None):
    """Convert trueform curves to a line-only PyVista PolyData.

    Accepts the ``(paths, points)`` pair every trueform curve producer
    returns — either as two arguments or as one tuple, so
    ``curves_to_pyvista(tf.isocontours(...))`` works directly. The paths
    feed ``vtkCellArray.SetData`` zero-copy, exactly like polygon faces.

    Parameters
    ----------
    paths : trueform.OffsetBlockedArray or (paths, points) tuple
        Polyline point indices, one block per curve.
    points : np.ndarray, optional
        ``(P, 3)`` float32/float64 curve points; omit when ``paths`` is the
        tuple.

    Returns
    -------
    pyvista.PolyData
    """
    if points is None and isinstance(paths, tuple) and len(paths) == 2:
        paths, points = paths
    if not isinstance(paths, tf.OffsetBlockedArray):
        raise TypeError(
            "paths must be a trueform.OffsetBlockedArray, "
            f"got {type(paths).__name__}")
    points = np.ascontiguousarray(_validated_points(points))
    result = pv.PolyData()
    result.SetPoints(pv.vtk_points(points, deep=False))
    result.SetLines(_cell_array(paths.offsets, paths.data))
    return result


__all__ = ["curves_to_pyvista", "to_pyvista", "to_trueform"]
