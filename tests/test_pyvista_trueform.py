"""
Tests for the PyVista integration

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import gc
import subprocess
import sys

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


def _two_cubes_concatenated():
    """One self-intersecting PolyData: two overlapping cubes, one soup."""
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    points = np.vstack([a.points, b.points])
    faces = np.vstack([a.regular_faces, b.regular_faces + a.n_points])
    return _polydata(points, [list(f) for f in faces])


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


# -- the MTime cache -----------------------------------------------------


def test_accessor_cache_reuses_one_mesh_instance():
    dataset = pv.Sphere()
    first = dataset.trueform.to_mesh()
    second = dataset.trueform.to_mesh()
    assert first is second

    # Operations consume the same cached instance, so its lazy structures
    # (tree, face membership, edge link) amortize across calls.
    assert dataset.trueform.is_closed()
    assert dataset.trueform.is_manifold()
    assert dataset.trueform.to_mesh() is first


def test_accessor_cache_survives_raw_mutation_until_modified():
    dataset = pv.Sphere()
    stale = dataset.trueform.to_mesh()

    # The documented VTK gotcha: mutating a raw NumPy view does not bump
    # the MTime, so the accessor keeps serving the stale mesh.
    raw = np.asarray(dataset.points)
    raw[0, 0] += 0.25
    assert dataset.trueform.to_mesh() is stale

    dataset.Modified()
    fresh = dataset.trueform.to_mesh()
    assert fresh is not stale
    assert fresh.points[0, 0] == raw[0, 0]


def test_accessor_cache_rebuilds_on_pyvista_edits():
    dataset = pv.Sphere()
    before = dataset.trueform.to_mesh()
    dataset.points = dataset.points * 2.0
    after = dataset.trueform.to_mesh()
    assert after is not before
    assert after.points.max() == pytest.approx(2.0 * before.points.max())


# -- accessor operations -------------------------------------------------


def test_accessor_booleans():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))

    union = a.trueform.union(b)
    assert union.volume == pytest.approx(1.875, rel=1e-4)
    assert len(union.cell_data["trueform_labels"]) == union.n_cells
    assert len(union.cell_data["trueform_face_labels"]) == union.n_cells
    assert union.trueform.is_closed()

    intersection = a.trueform.intersection(b)
    assert intersection.volume == pytest.approx(0.125, rel=1e-4)

    difference = a.trueform.difference(b)
    assert difference.volume == pytest.approx(0.875, rel=1e-4)

    reverse = b.trueform.difference(a)
    assert reverse.volume == pytest.approx(0.875, rel=1e-4)


def test_accessor_boolean_curves():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    result, curves = a.trueform.union(b, return_curves=True)
    assert result.n_cells > 0
    assert curves.GetNumberOfLines() > 0


def test_accessor_intersection_curves():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    curves = a.trueform.intersection_curves(b)
    assert curves.GetNumberOfLines() > 0
    assert curves.n_points > 0


def test_accessor_self_intersection_family():
    soup = _two_cubes_concatenated()

    curves = soup.trueform.self_intersection_curves()
    assert curves.GetNumberOfLines() > 0

    arranged = soup.trueform.polygon_arrangements()
    assert arranged.n_cells > soup.n_cells
    assert len(arranged.cell_data["trueform_face_labels"]) == arranged.n_cells

    shell = soup.trueform.outer_shell()
    assert shell.trueform.is_closed()
    assert shell.volume == pytest.approx(1.875, rel=1e-4)


def test_accessor_isocontours_and_isobands():
    sphere = pv.Sphere()
    sphere.point_data["height"] = np.asarray(sphere.points)[:, 2].copy()

    by_name = sphere.trueform.isocontours("height", 0.0)
    assert by_name.GetNumberOfLines() > 0

    by_array = sphere.trueform.isocontours(
        np.asarray(sphere.points)[:, 2], [-0.2, 0.0, 0.2])
    assert by_array.GetNumberOfLines() >= 3

    bands = sphere.trueform.isobands("height", [0.0])
    assert bands.n_cells > 0
    labels = bands.cell_data["trueform_labels"]
    assert set(np.unique(labels)) == {0, 1}
    assert len(bands.cell_data["trueform_face_labels"]) == bands.n_cells

    upper, curves = sphere.trueform.isobands(
        "height", [0.0], selected_bands=[1], return_curves=True)
    assert set(np.unique(upper.cell_data["trueform_labels"])) == {1}
    assert curves.GetNumberOfLines() > 0


def test_accessor_cleaned():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0],
         [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # two vertices duplicated
        dtype=np.float32)
    soup = _polydata(points, [[0, 1, 2], [3, 4, 5]])
    merged = soup.trueform.cleaned()
    assert merged.n_points == 4
    assert merged.n_cells == 2


def test_accessor_triangulated():
    grid = pv.Plane(i_resolution=2, j_resolution=2)  # quads
    assert not grid.is_all_triangles
    triangles = grid.trueform.triangulated()
    assert triangles.is_all_triangles
    assert triangles.n_cells == 8


def test_accessor_diagnostics():
    assert _cube().trueform.is_closed()
    assert _cube().trueform.is_manifold()
    plane = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    assert not plane.trueform.is_closed()
    assert plane.trueform.is_manifold()


# -- packaging -----------------------------------------------------------


def test_entry_point_registers_accessor_without_import():
    if not hasattr(pv, "register_dataset_accessor"):
        pytest.skip("pyvista without accessor registry")
    code = (
        "import pyvista as pv\n"
        "sphere = pv.Sphere()\n"
        "assert sphere.trueform.is_closed()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
