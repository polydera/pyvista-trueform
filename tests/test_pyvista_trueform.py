"""
Tests for the PyVista integration

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import gc

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

import trueform as tf
import pyvista_trueform as tfpv
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkCellArray

POINT_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]


def _polydata(points, polygons, *, index_dtype=np.int64):
    """Polygon-only PolyData with an explicit cell-array index dtype."""
    points = np.ascontiguousarray(points)
    offsets = np.zeros(len(polygons) + 1, dtype=index_dtype)
    if polygons:
        sizes = np.fromiter((len(cell) for cell in polygons),
                            dtype=index_dtype, count=len(polygons))
        np.cumsum(sizes, out=offsets[1:])
        connectivity = np.concatenate(
            [np.asarray(cell, dtype=index_dtype) for cell in polygons])
    else:
        connectivity = np.empty(0, dtype=index_dtype)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtk(offsets, deep=True),
                  numpy_to_vtk(connectivity, deep=True))
    result = pv.PolyData()
    result.SetPoints(pv.vtk_points(points, deep=True))
    result.SetPolys(cells)
    return result


def _cube(center=(0.0, 0.0, 0.0)):
    return pv.Cube(center=center).triangulate()


# -- conversions ---------------------------------------------------------


@pytest.mark.parametrize("point_dtype", POINT_DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_round_trip_fixed(point_dtype, index_dtype):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=point_dtype)
    source = _polydata(points, [[0, 1, 2], [1, 3, 2]],
                       index_dtype=index_dtype)

    mesh = tfpv.to_trueform(source)
    assert isinstance(mesh, tf.Mesh)
    assert isinstance(mesh.faces, np.ndarray)
    assert mesh.faces.dtype == index_dtype
    assert mesh.points.dtype == point_dtype
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(mesh.points, points)

    back = tfpv.to_pyvista(mesh)
    assert back.n_cells == 2
    assert back.is_all_triangles
    np.testing.assert_array_equal(back.regular_faces, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(np.asarray(back.points), points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_round_trip_dynamic(index_dtype):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]],
        dtype=np.float32)
    source = _polydata(points, [[0, 1, 2, 3], [1, 4, 2]],
                       index_dtype=index_dtype)

    mesh = tfpv.to_trueform(source)
    assert isinstance(mesh.faces, tf.OffsetBlockedArray)
    assert mesh.faces.data.dtype == index_dtype
    np.testing.assert_array_equal(mesh.faces.offsets, [0, 4, 7])
    np.testing.assert_array_equal(mesh.faces.data, [0, 1, 2, 3, 1, 4, 2])

    back = tfpv.to_pyvista(mesh)
    assert back.n_cells == 2
    assert not back.is_all_triangles
    np.testing.assert_array_equal(
        vtk_to_numpy(back.GetPolys().GetOffsetsArray()), [0, 4, 7])
    np.testing.assert_array_equal(
        vtk_to_numpy(back.GetPolys().GetConnectivityArray()),
        [0, 1, 2, 3, 1, 4, 2])
    np.testing.assert_array_equal(np.asarray(back.points), points)


def test_to_trueform_is_detached():
    source = _cube()
    mesh = tfpv.to_trueform(source)
    before = mesh.points.copy()
    source.points = source.points + 1.0
    np.testing.assert_array_equal(mesh.points, before)


def test_to_trueform_refuses_non_polygon_cells():
    with pytest.raises(ValueError, match="polygons only"):
        tfpv.to_trueform(pv.Line())
    with pytest.raises(TypeError):
        tfpv.to_trueform(object())


def test_to_pyvista_fixed_is_zero_copy_and_owner_safe():
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    result = tfpv.to_pyvista((faces, points))

    assert np.shares_memory(np.asarray(result.points), points)
    assert np.shares_memory(
        vtk_to_numpy(result.GetPolys().GetConnectivityArray()), faces)

    del faces, points
    gc.collect()
    assert result.n_cells == 2
    np.testing.assert_array_equal(result.regular_faces, [[0, 1, 2], [1, 3, 2]])


def test_to_pyvista_dynamic_is_zero_copy_and_owner_safe():
    faces = tf.OffsetBlockedArray(
        np.array([0, 4, 7], dtype=np.int32),
        np.array([0, 1, 2, 3, 1, 4, 2], dtype=np.int32))
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0]],
        dtype=np.float64)
    result = tfpv.to_pyvista((faces, points))

    assert np.shares_memory(np.asarray(result.points), points)
    assert np.shares_memory(
        vtk_to_numpy(result.GetPolys().GetConnectivityArray()), faces.data)
    assert np.shares_memory(
        vtk_to_numpy(result.GetPolys().GetOffsetsArray()), faces.offsets)

    del faces, points
    gc.collect()
    assert result.n_cells == 2
    np.testing.assert_array_equal(
        vtk_to_numpy(result.GetPolys().GetConnectivityArray()),
        [0, 1, 2, 3, 1, 4, 2])


def test_to_pyvista_applies_mesh_transformation():
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, 3] = [5, 0, 0]
    mesh = tf.Mesh(faces, points, transformation=transformation)

    moved = tfpv.to_pyvista(mesh)
    np.testing.assert_allclose(
        np.asarray(moved.points), points + [5, 0, 0], atol=1e-6)

    local = tfpv.to_pyvista(mesh, apply_transformation=False)
    np.testing.assert_array_equal(np.asarray(local.points), points)


def test_to_pyvista_empty_mesh():
    result = tfpv.to_pyvista(
        (np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)))
    assert result.n_cells == 0
    assert result.n_points == 0


def test_to_pyvista_rejects_bad_input():
    with pytest.raises(TypeError):
        tfpv.to_pyvista(object())
    with pytest.raises(ValueError):
        tfpv.to_pyvista((np.array([[0, 1, 2, 3]], dtype=np.int32),
                         np.empty((4, 3), dtype=np.float32)))


def test_curves_to_pyvista_two_arguments_and_tuple():
    paths = tf.OffsetBlockedArray(
        np.array([0, 3, 5], dtype=np.int32),
        np.array([0, 1, 2, 3, 4], dtype=np.int32))
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]],
        dtype=np.float32)

    curves = tfpv.curves_to_pyvista(paths, points)
    assert curves.n_cells == 2
    assert curves.GetNumberOfLines() == 2
    assert np.shares_memory(np.asarray(curves.points), points)
    np.testing.assert_array_equal(
        vtk_to_numpy(curves.GetLines().GetConnectivityArray()),
        [0, 1, 2, 3, 4])

    as_tuple = tfpv.curves_to_pyvista((paths, points))
    assert as_tuple.n_cells == 2

