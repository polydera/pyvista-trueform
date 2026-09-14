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
_SAMPLES = "trueform_samples"


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


def _validated_image(dataset):
    if not isinstance(dataset, pv.ImageData):
        raise TypeError(
            "dataset must be a pyvista.ImageData, got "
            f"{type(dataset).__name__}")
    return dataset


def _field_array(dataset, scalars):
    """The single-component point-data array a volume conversion reads."""
    if scalars is None:
        scalars = dataset.point_data.active_scalars_name
        if scalars is None:
            carried = ", ".join(
                repr(name) for name in dataset.cell_data.keys())
            raise ValueError(
                "ImageData has no active point-data scalars; a volume's "
                "samples are its grid nodes. Name a point array with "
                "scalars=, or set one active with "
                "dataset.set_active_scalars(name, preference='point')"
                + (f". This dataset carries cell data only ({carried}); "
                   "dataset.cell_data_to_point_data() moves it to the nodes"
                   if carried else ""))
    if scalars not in dataset.point_data.keys():
        carried = ", ".join(repr(name) for name in dataset.point_data.keys())
        raise ValueError(
            f"ImageData has no point-data array {scalars!r}; it carries "
            + (carried or "none"))
    values = dataset.point_data[scalars]
    if values.ndim != 1:
        raise ValueError(
            "samples must be a single-component point-data array; "
            f"{scalars!r} has shape {values.shape}")
    return values


def _index_to_physical(volume):
    """Where a trueform Volume places voxel index ``(i, j, k)``, as one 4x4.

    The grid's own ``origin + index * spacing``, composed under the world
    pose when the volume carries one.
    """
    placement = np.eye(4)
    placement[:3, :3] = np.diag(volume.spacing)
    placement[:3, 3] = volume.origin
    if volume.transformation is None:
        return placement
    return np.asarray(volume.transformation, dtype=np.float64) @ placement


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
    # VTK reads the cell count off the offsets array, which holds one entry
    # per cell plus the terminator; an empty producer has no cells to state
    # and hands over nothing, which VTK would read as a count of -1.
    if len(offsets) == 0:
        offsets = np.zeros(1, dtype=np.asarray(connectivity).dtype)
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtk(np.ascontiguousarray(offsets), deep=False),
        numpy_to_vtk(np.ascontiguousarray(connectivity), deep=False),
    )
    return cells


def _line_paths(dataset):
    """Detached trueform paths of a line-only PolyData: unordered 2-point
    segments connect into polylines through
    :func:`trueform.connect_edges_to_paths`; polyline cells pass through
    (copied, so the paths never alias the dataset's VTK arrays).
    """
    if not isinstance(dataset, pv.PolyData):
        raise TypeError(
            f"lines must be a pyvista.PolyData, got {type(dataset).__name__}")
    families = {
        "vertices": dataset.GetNumberOfVerts(),
        "polygons": dataset.GetNumberOfPolys(),
        "strips": dataset.GetNumberOfStrips(),
    }
    present = [name for name, count in families.items() if count]
    if present:
        raise ValueError(
            "PolyData must contain lines only; found " + ", ".join(present))
    if not dataset.GetNumberOfLines():
        raise ValueError("PolyData contains no lines")
    cells = dataset.GetLines()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    if np.all(np.diff(offsets) == 2):
        return tf.connect_edges_to_paths(
            np.ascontiguousarray(connectivity.reshape(-1, 2)))
    return tf.OffsetBlockedArray(np.array(offsets, copy=True, order="C"),
                                 np.array(connectivity, copy=True, order="C"))


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


def domains_to_pyvista(cells, ids):
    """Assemble trueform domain cells into a named PyVista MultiBlock.

    Accepts the ``(cells, ids)`` pair :meth:`trueform.CsgGraph.domains`
    returns — either as two arguments' worth in one tuple or separately.
    Each domain mesh converts through :func:`to_pyvista` zero-copy; block
    ``k`` is named ``str(ids[k])``.

    Parameters
    ----------
    cells : list of (faces, points)
        One mesh per kept domain.
    ids : np.ndarray
        ``ids[k]`` is the domain id of cell ``k``.

    Returns
    -------
    pyvista.MultiBlock
    """
    if len(cells) != len(ids):
        raise ValueError(
            f"cells and ids must have equal length, got {len(cells)} cells "
            f"and {len(ids)} ids")
    result = pv.MultiBlock()
    for cell, domain_id in zip(cells, ids):
        result.append(to_pyvista(cell), str(domain_id))
    return result


def volume_to_trueform(dataset, scalars=None):
    """Copy a PyVista ImageData's scalar field into a fresh trueform Volume.

    The samples cross as they stand: VTK's point-data order and trueform's
    voxel order are the same flat x-fastest run, so the field is one
    reshape, and the dtype is the array's own (trueform carries float32,
    float64, int16, uint16 and uint8 fields, and converts anything else to
    float32 — an int16 CT stays int16 on its float32 grid).

    The grid crosses as the placement it is: ``spacing``, ``dimensions``
    and ``origin`` pass through, and the direction matrix becomes the
    volume's 4x4 :attr:`trueform.Volume.transformation` — a turn about
    the dataset's own origin, so the volume's local space IS the
    dataset's own axis-aligned space and a positional input stays
    readable in the coordinates the dataset states. An identity
    direction composes to exactly the identity, which trueform reads as
    unposed. A dataset whose extent does not start at zero — an
    ``extract_subset`` that kept its coordinates, a ``.vti`` with a
    nonzero ``WholeExtent`` — carries that offset in its origin, since
    the volume's first sample is the first one stored.

    The volume is detached: the samples are copied, so later PyVista edits
    do not affect it. Retain it — or use the ``dataset.trueform``
    accessor, which caches exactly this conversion.

    Parameters
    ----------
    dataset : pyvista.ImageData
        The grid carrying the field as point data.
    scalars : str, optional
        Which point-data array is the field. Default: the dataset's active
        point scalars.

    Returns
    -------
    trueform.Volume
    """
    _validated_image(dataset)
    values = _field_array(dataset, scalars)
    samples = np.array(values.reshape(dataset.dimensions, order="F"),
                       order="F", copy=True)
    direction = np.asarray(dataset.direction_matrix)
    stated = np.asarray(dataset.origin)
    pose = np.eye(4)
    pose[:3, :3] = direction
    pose[:3, 3] = stated - direction @ stated
    volume = tf.Volume(
        samples, spacing=dataset.spacing,
        origin=stated + np.asarray(dataset.extent[::2]) * dataset.spacing)
    volume.transformation = pose.astype(volume.coordinate_dtype)
    return volume


def volume_to_pyvista(volume):
    """Convert a trueform Volume to a fresh PyVista ImageData.

    Zero-copy: the samples ride into VTK as the flat x-fastest array they
    already are, under the name ``trueform_samples`` (the grid's active
    point scalars), and VTK retains the NumPy buffer.

    A posed volume needs no baking — unlike a :class:`trueform.Mesh`
    transformation, which :func:`to_pyvista` must bake into the points, a
    volume's world pose is carried by the grid itself: the placement
    ``transformation @ (origin + index * spacing)`` is handed to
    :attr:`pyvista.ImageData.index_to_physical_matrix`, which splits it
    into ``origin``, ``spacing`` and ``direction_matrix``. A pose that
    shears the grid has no such split and PyVista refuses it there.

    Parameters
    ----------
    volume : trueform.Volume
        The field.

    Returns
    -------
    pyvista.ImageData
    """
    if not isinstance(volume, tf.Volume):
        raise TypeError(
            f"volume must be a trueform.Volume, got {type(volume).__name__}")
    result = pv.ImageData(dimensions=volume.dims)
    result.index_to_physical_matrix = _index_to_physical(volume)
    result.point_data[_SAMPLES] = volume.samples.ravel(order="F")
    return result


__all__ = ["curves_to_pyvista", "domains_to_pyvista", "to_pyvista",
           "to_trueform", "volume_to_pyvista", "volume_to_trueform"]
