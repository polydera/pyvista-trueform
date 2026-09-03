"""
Tests for the PyVista integration

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import gc
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

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


def _ray(origin, direction):
    """A float32 trueform Ray, matching PyVista's float32 points."""
    return tf.Ray(origin=np.asarray(origin, dtype=np.float32),
                  direction=np.asarray(direction, dtype=np.float32))


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


def test_accessor_domains_self_decomposition():
    soup = _two_cubes_concatenated()
    blocks = soup.trueform.domains()
    assert isinstance(blocks, pv.MultiBlock)
    assert blocks.n_blocks == 3
    volumes = sorted(block.volume for block in blocks)
    np.testing.assert_allclose(volumes, [0.125, 0.875, 0.875], rtol=1e-4)
    for block in blocks:
        assert block.trueform.is_closed()


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


# -- remesh --------------------------------------------------------------


def test_accessor_remeshed():
    sphere = pv.Sphere()
    target = 2.0 * tf.mean_edge_length(sphere.trueform.to_mesh())

    remeshed = sphere.trueform.remeshed(target)
    result_length = tf.mean_edge_length(remeshed.trueform.to_mesh())
    assert 0.5 * target < result_length < 1.5 * target
    assert remeshed.trueform.is_closed()


def test_accessor_decimated():
    sphere = pv.Sphere()
    decimated = sphere.trueform.decimated(0.3)
    assert 0 < decimated.n_cells <= 0.3 * sphere.n_cells
    assert decimated.trueform.is_closed()


def test_accessor_simplified():
    sphere = pv.Sphere()
    simplified = sphere.trueform.simplified()
    assert 0 < simplified.n_cells < sphere.n_cells
    assert simplified.trueform.is_closed()


def test_accessor_remesh_region_labels_ride():
    sphere = pv.Sphere()
    regions = (np.asarray(sphere.points)[sphere.regular_faces[:, 0], 2]
               > 0).astype(np.int32)
    remeshed = sphere.trueform.remeshed(
        2.0 * tf.mean_edge_length(sphere.trueform.to_mesh()),
        preserve_regions=regions)
    labels = remeshed.cell_data["trueform_labels"]
    assert len(labels) == remeshed.n_cells
    assert set(np.unique(labels)) == {0, 1}


# -- queries and measurements --------------------------------------------


def test_accessor_measurements_exact():
    cube = _cube()
    assert cube.trueform.volume() == 1.0
    assert cube.trueform.area() == 6.0


def test_accessor_euler_characteristic():
    sphere = pv.Sphere().triangulate()
    assert sphere.trueform.euler_characteristic() == 2

    plane = pv.Plane(i_resolution=3, j_resolution=3).triangulate()
    assert plane.trueform.euler_characteristic() == 1

    torus = pv.ParametricTorus(u_res=10, v_res=10).triangulate()
    assert torus.trueform.euler_characteristic() == 0


def test_accessor_signed_distance_known_values():
    cube = _cube()  # extent [-0.5, 0.5], float32
    queries = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64))

    result = queries.trueform.signed_distance(cube)
    np.testing.assert_allclose(result, [-0.5, 1.5])
    assert result.dtype == cube.points.dtype

    # A trueform.Mesh target works the same as a PolyData one.
    mesh_result = queries.trueform.signed_distance(cube.trueform.to_mesh())
    np.testing.assert_array_equal(mesh_result, result)


def test_accessor_ray_cast_known_hit():
    cube = _cube(center=(3.0, 0.0, 0.0))  # near face at x = 2.5
    face_id, t = cube.trueform.ray_cast(
        _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    assert t == 2.5
    hit_face = cube.regular_faces[face_id]
    np.testing.assert_array_equal(np.asarray(cube.points)[hit_face, 0],
                                  [2.5, 2.5, 2.5])
    assert cube.trueform.ray_cast(
        _ray([0.0, 0.0, 5.0], [1.0, 0.0, 0.0])) is None
    # config is the (min_t, max_t) range, forwarded verbatim
    assert cube.trueform.ray_cast(
        _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), config=(0.0, 2.0)) is None
    with pytest.raises(TypeError, match="trueform.Ray"):
        cube.trueform.ray_cast([0.0, 0.0, 0.0])


def test_accessor_ray_cast_batch():
    cube = _cube(center=(3.0, 0.0, 0.0))  # near face at x = 2.5
    rays = _ray([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert rays.is_batch
    face_ids, ts = cube.trueform.ray_cast(rays)
    assert face_ids.shape == (2,)
    assert ts[0] == 2.5
    np.testing.assert_array_equal(
        np.asarray(cube.points)[cube.regular_faces[face_ids[0]], 0],
        [2.5, 2.5, 2.5])
    assert face_ids[1] == -1
    assert np.isnan(ts[1])


def test_accessor_distance_and_intersects():
    a = _cube()
    far = _cube(center=(3.0, 0.0, 0.0))  # gap from x=0.5 to x=2.5
    assert a.trueform.distance(far) == 2.0
    assert not a.trueform.intersects(far)

    near = _cube(center=(0.5, 0.5, 0.5))
    assert a.trueform.distance(near) == 0.0
    assert a.trueform.intersects(near)

    assert a.trueform.distance([2.5, 0.0, 0.0]) == 2.0


def test_accessor_closest_point_on_corner():
    cube = _cube()
    face_id, distance, point = cube.trueform.closest_point([2.0, 2.0, 2.0])
    # euclidean to the (0.5, 0.5, 0.5) corner: sqrt(3 * 1.5**2)
    assert distance == math.sqrt(6.75)
    np.testing.assert_array_equal(point, [0.5, 0.5, 0.5])
    assert (cube.regular_faces[face_id]
            == cube.find_closest_point([0.5, 0.5, 0.5])).any()


def test_accessor_closest_point_pair_between_meshes():
    a = _cube()
    far = _cube(center=(3.0, 0.0, 0.0))  # gap from x=0.5 to x=2.5
    (face_a, face_far), (distance, point_a, point_far) = \
        a.trueform.closest_point_pair(far)
    assert distance == 2.0
    assert point_a[0] == 0.5
    assert point_far[0] == 2.5
    np.testing.assert_array_equal(point_a[1:], point_far[1:])
    np.testing.assert_array_equal(
        np.asarray(a.points)[a.regular_faces[face_a], 0], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(
        np.asarray(far.points)[far.regular_faces[face_far], 0],
        [2.5, 2.5, 2.5])

    mesh_pair = a.trueform.closest_point_pair(far.trueform.to_mesh())
    assert mesh_pair[1][0] == 2.0


def test_accessor_closest_point_pair_with_bare_points():
    cube = _cube()
    operand = pv.PolyData(np.array(
        [[2.5, 0.0, 0.0], [5.0, 5.0, 5.0], [-3.0, 0.0, 0.0]],
        dtype=np.float32))
    assert operand.GetNumberOfPolys() == 0
    (face, point_id), (distance, point, other_point) = \
        cube.trueform.closest_point_pair(operand)
    assert point_id == 0
    assert distance == 2.0
    np.testing.assert_array_equal(point, [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(other_point, [2.5, 0.0, 0.0])
    np.testing.assert_array_equal(
        np.asarray(cube.points)[cube.regular_faces[face], 0],
        [0.5, 0.5, 0.5])


def test_accessor_closest_point_pair_refuses_pointless_operand():
    with pytest.raises(TypeError, match="operand must be"):
        _cube().trueform.closest_point_pair(pv.MultiBlock())


def test_accessor_curvatures_and_shape_index():
    sphere = pv.Sphere()  # radius 0.5, curvature 1/r = 2
    k0, k1 = sphere.trueform.principal_curvatures()
    assert k0.shape == (sphere.n_points,)
    assert 1.5 < k0.mean() < 2.5
    assert 1.5 < k1.mean() < 2.5

    shape_index = sphere.trueform.shape_index()  # spherical cap = +1
    assert shape_index.shape == (sphere.n_points,)
    assert shape_index.mean() > 0.9


def test_accessor_boundary_curves_of_plane():
    plane = pv.Plane(i_resolution=1, j_resolution=1).triangulate()
    curves = plane.trueform.boundary_curves()
    assert curves.GetNumberOfLines() == 1
    assert curves.n_points == 4

    path = vtk_to_numpy(curves.GetLines().GetConnectivityArray())
    assert path[0] == path[-1]  # one closed loop
    steps = np.diff(np.asarray(curves.points)[path], axis=0)
    assert np.linalg.norm(steps, axis=1).sum() == 4.0  # unit-square rim


# -- io ------------------------------------------------------------------


def test_io_round_trip_stl(tmp_path):
    cube = _cube()
    path = tmp_path / "cube.stl"
    tfpv.write(path, cube)

    back = tfpv.read(path)
    assert back.n_cells == 12
    # STL stores bare triangles; trueform welds duplicates on read, so the
    # cube's 8 corners come back exactly, in some order.
    assert back.n_points == 8
    np.testing.assert_array_equal(
        np.sort(np.asarray(back.points), axis=0),
        np.sort(np.asarray(cube.points), axis=0))
    assert back.volume == pytest.approx(1.0)
    assert back.trueform.is_closed()


def test_io_round_trip_obj(tmp_path):
    grid = pv.Plane(i_resolution=2, j_resolution=2)  # quads survive OBJ
    path = tmp_path / "grid.obj"
    tfpv.write(path, grid)

    back = tfpv.read(path)
    assert back.n_cells == 4
    assert back.n_points == 9
    np.testing.assert_array_equal(
        np.asarray(back.points), np.asarray(grid.points))
    np.testing.assert_array_equal(
        vtk_to_numpy(back.GetPolys().GetConnectivityArray()),
        vtk_to_numpy(grid.GetPolys().GetConnectivityArray()))


def test_io_refuses_unknown_suffix(tmp_path):
    with pytest.raises(ValueError, match=r"\.ply"):
        tfpv.read(tmp_path / "mesh.ply")
    with pytest.raises(ValueError, match=r"\.ply"):
        tfpv.write(tmp_path / "mesh.ply", _cube())


# -- the N-ary factory ---------------------------------------------------


def test_csg_graph_end_to_end():
    a = _cube()
    b = _cube(center=(0.5, 0.0, 0.0))
    c = _cube(center=(0.25, 0.5, 0.0))
    graph = tfpv.csg_graph([a, b, c])
    assert isinstance(graph, tfpv.CsgGraph)

    difference = graph.mesh(tf.op(0) - tf.op(1))
    assert isinstance(difference, pv.PolyData)
    assert difference.n_cells > 0
    assert difference.trueform.is_closed()
    assert difference.volume == pytest.approx(0.5, rel=1e-4)

    union = graph.mesh(tf.op(0) | tf.op(1) | tf.op(2))
    assert isinstance(union, pv.PolyData)
    assert union.trueform.is_closed()
    assert union.volume > 1.5

    full = graph.mesh()
    assert full.n_cells >= union.n_cells


def test_csg_graph_refuses_quads_with_pyvista_wording():
    quads = pv.Plane(i_resolution=2, j_resolution=2)
    with pytest.raises(ValueError, match=r"\.triangulate\(\)"):
        tfpv.csg_graph([quads, _cube()])


def test_csg_graph_operands_reuse_accessor_cache():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    graph = tfpv.csg_graph([a, b])
    assert graph.native.forms[0] is a.trueform.to_mesh()
    assert graph.native.forms[1] is b.trueform.to_mesh()


def test_csg_graph_readers_answer_in_pyvista_types():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    graph = tfpv.csg_graph([a, b])

    blocks = graph.domains()
    assert isinstance(blocks, pv.MultiBlock)
    module_blocks = tfpv.domains(graph)
    assert blocks.keys() == module_blocks.keys()
    assert [block.n_cells for block in blocks] == [
        block.n_cells for block in module_blocks]

    curves = graph.intersection_curves()
    assert isinstance(curves, pv.PolyData)
    assert curves.GetNumberOfLines() > 0
    assert curves.GetNumberOfPolys() == 0


def test_csg_graph_outer_shell():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    graph = tfpv.csg_graph([a, b])

    shell = graph.outer_shell()
    assert isinstance(shell, pv.PolyData)
    assert shell.trueform.is_closed()
    union_volume = graph.mesh(tf.op(0) | tf.op(1)).volume
    assert union_volume == pytest.approx(1.875, rel=1e-4)
    assert shell.volume == pytest.approx(union_volume, rel=1e-4)


# -- arrangements and domains --------------------------------------------


def test_mesh_arrangements_labeled():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    arranged = tfpv.mesh_arrangements([a, b])
    assert arranged.n_cells > a.n_cells + b.n_cells
    assert set(np.unique(arranged.cell_data["trueform_labels"])) == {0, 1}
    assert len(arranged.cell_data["trueform_face_labels"]) == arranged.n_cells

    with_curves, curves = tfpv.mesh_arrangements([a, b], return_curves=True)
    assert with_curves.n_cells == arranged.n_cells
    assert curves.GetNumberOfLines() > 0


def test_domains_of_two_cubes():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    blocks = tfpv.domains([a, b])
    assert isinstance(blocks, pv.MultiBlock)
    assert blocks.n_blocks == 3  # A-only, B-only, and the shared core

    union_volume = a.trueform.union(b).volume
    assert sum(block.volume for block in blocks) == pytest.approx(
        union_volume, rel=1e-4)
    for block in blocks:
        assert block.trueform.is_closed()

    graph = tfpv.csg_graph([a, b])
    core = tfpv.domains(graph, tf.op(0) & tf.op(1))
    assert core.n_blocks == 1
    assert core[0].volume == pytest.approx(0.125, rel=1e-4)
    assert core.keys()[0] == str(
        graph.native.domains(tf.op(0) & tf.op(1))[1][0])


def test_domains_block_names_are_domain_ids():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    graph = tfpv.csg_graph([a, b])
    cells, ids = graph.native.domains()
    assert tfpv.domains(graph).keys() == [str(domain_id) for domain_id in ids]

    blocks = tfpv.domains_to_pyvista(cells, ids)  # zero-copy assembly
    assert blocks.keys() == [str(domain_id) for domain_id in ids]
    for block, (faces, points) in zip(blocks, cells):
        assert block.n_cells == len(faces)
        assert np.shares_memory(np.asarray(block.points), points)


# -- picking -------------------------------------------------------------


def _two_far_cubes_multiblock():
    return pv.MultiBlock([_cube(), _cube(center=(3.0, 0.0, 0.0))])


def test_pick_names_the_domain_block():
    a = _cube()                        # extent [-0.5, 0.5]
    b = _cube(center=(0.5, 0.5, 0.5))  # extent [0, 1]
    blocks = tfpv.domains(tfpv.csg_graph([a, b]))
    hit = tfpv.pick(blocks, _ray([5.0, 0.25, 0.25], [-1.0, 0.0, 0.0]))
    # first surface on the way: b's +x face at x = 1, on the B-only domain
    expected = next(k for k in range(blocks.n_blocks)
                    if blocks[k].bounds[1] == 1.0)
    assert hit.block_index == expected
    assert hit.block is blocks[expected]
    assert hit.t == 4.0
    np.testing.assert_array_equal(hit.point, [1.0, 0.25, 0.25])
    np.testing.assert_array_equal(
        np.asarray(hit.block.points)[hit.block.regular_faces[hit.face], 0],
        [1.0, 1.0, 1.0])

    assert tfpv.pick(blocks, _ray([5.0, 0.25, 0.25], [1.0, 0.0, 0.0])) is None


def test_pick_nested_multiblock_flat_indexing():
    a = _cube()
    b = _cube(center=(0.5, 0.5, 0.5))
    blocks = tfpv.domains(tfpv.csg_graph([a, b]))
    ray = _ray([5.0, 0.25, 0.25], [-1.0, 0.0, 0.0])
    flat = tfpv.pick(blocks, ray)
    nested = tfpv.pick(pv.MultiBlock([blocks]), ray)
    assert nested.block_index == flat.block_index
    assert nested.face == flat.face
    assert nested.t == flat.t
    np.testing.assert_array_equal(nested.point, flat.point)


def test_pick_plain_polydata_target():
    cube = _cube(center=(3.0, 0.0, 0.0))
    hit = tfpv.pick(cube, _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    assert hit.block_index == 0
    assert hit.block is cube
    assert hit.t == 2.5
    np.testing.assert_array_equal(hit.point, [2.5, 0.0, 0.0])


def test_pick_skips_none_blocks_and_keeps_their_numbers():
    blocks = pv.MultiBlock()
    blocks.append(None)
    blocks.append(_cube(center=(3.0, 0.0, 0.0)))
    hit = tfpv.pick(blocks, _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    assert hit.block_index == 1
    assert hit.t == 2.5


def test_pick_refuses_non_polydata_block():
    blocks = pv.MultiBlock([_cube(), pv.ImageData()])
    with pytest.raises(TypeError, match="block 1 must be"):
        tfpv.pick(blocks, _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
    with pytest.raises(TypeError, match="target must be"):
        tfpv.pick(42, _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]))


def test_pick_refuses_batch_and_non_ray():
    with pytest.raises(TypeError, match="trueform.Ray"):
        tfpv.pick(_cube(), [0.0, 0.0, 0.0])
    batch = _ray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                 [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="single ray"):
        tfpv.pick(_cube(), batch)


def test_closest_point_query_names_the_block():
    hit = tfpv.closest(_two_far_cubes_multiblock(), [2.0, 0.0, 0.0])
    assert hit.block_index == 1
    assert hit.distance == 0.5
    np.testing.assert_array_equal(hit.point, [2.5, 0.0, 0.0])


def test_closest_mesh_query_gap():
    probe = pv.Cube(center=(4.0, 0.0, 0.0), x_length=0.5, y_length=0.5,
                    z_length=0.5).triangulate()
    hit = tfpv.closest(_two_far_cubes_multiblock(), probe)
    assert hit.block_index == 1
    assert hit.distance == 0.25  # gap from x=3.5 to the probe at x=3.75
    assert hit.point[0] == 3.5


def test_closest_bare_points_query_names_the_block():
    probe = pv.PolyData(np.array(
        [[2.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32))
    hit = tfpv.closest(_two_far_cubes_multiblock(), probe)
    assert hit.block_index == 1
    assert hit.distance == 0.5
    np.testing.assert_array_equal(hit.point, [2.5, 0.0, 0.0])


def test_closest_plain_polydata_target():
    cube = _cube()
    hit = tfpv.closest(cube, [1.0, 0.0, 0.0])
    assert hit.block_index == 0
    assert hit.block is cube
    assert hit.distance == 0.5
    np.testing.assert_array_equal(hit.point, [0.5, 0.0, 0.0])


def test_closest_empty_multiblock():
    assert tfpv.closest(pv.MultiBlock(), [0.0, 0.0, 0.0]) is None


# -- registration ---------------------------------------------------------


def _rotation_translation(angle_degrees, translation, dtype):
    angle = np.radians(angle_degrees)
    matrix = np.eye(4, dtype=dtype)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]
    matrix[:3, 3] = translation
    return matrix


def _applied(matrix, points):
    return points @ matrix[:3, :3].T.astype(points.dtype) \
        + matrix[:3, 3].astype(points.dtype)


def _elongated_cloud():
    rng = np.random.default_rng(7)
    return (rng.random((300, 3)) *
            [4.0, 1.0, 0.5]).astype(np.float32)


def test_align_rigid_recovers_known_transform():
    sphere = pv.Sphere()
    truth = _rotation_translation(30.0, [0.3, -0.2, 0.5], np.float32)
    moved = sphere.copy()
    moved.points = _applied(truth, np.asarray(sphere.points))

    recovered = tfpv.align(moved, sphere, method="rigid")
    assert recovered.shape == (4, 4)
    # Corresponding points, so the recovered delta is the exact inverse.
    np.testing.assert_allclose(
        _applied(recovered, np.asarray(moved.points)),
        np.asarray(sphere.points), atol=1e-4)


def test_align_similarity_recovers_known_scale():
    sphere = pv.Sphere()
    truth = _rotation_translation(30.0, [0.3, -0.2, 0.5], np.float32)
    scale = 1.7
    truth[:3, :3] *= scale
    moved = sphere.copy()
    moved.points = _applied(truth, np.asarray(sphere.points))

    recovered = tfpv.align(moved, sphere, method="similarity")
    assert recovered.shape == (4, 4)
    recovered_scale = np.linalg.det(recovered[:3, :3]) ** (1 / 3)
    assert recovered_scale == pytest.approx(1.0 / scale, rel=1e-4)
    np.testing.assert_allclose(
        _applied(recovered, np.asarray(moved.points)),
        np.asarray(sphere.points), atol=1e-4)


@pytest.mark.parametrize("method,tolerance", [("icp", 0.01), ("obb", 1e-4),
                                              ("knn", 0.01)])
def test_align_correspondence_free_methods(method, tolerance):
    points = _elongated_cloud()
    degrees, shift = (20.0, [0.4, -0.3, 0.2])
    if method == "knn":  # one soft-correspondence step: small motion only
        degrees, shift = (0.0, [0.05, 0.02, 0.03])
    truth = _rotation_translation(degrees, shift, np.float32)
    moved = _applied(truth, points)

    recovered = tfpv.align(moved, points, method=method)
    assert tfpv.chamfer_distance(
        _applied(recovered, moved), points) < tolerance


def test_align_refuses_unknown_method():
    with pytest.raises(ValueError, match="supported"):
        tfpv.align(pv.Sphere(), pv.Sphere(), method="banana")


def test_chamfer_distance_exact_shift():
    cube = _cube()
    shifted = cube.copy()
    shifted.points = (np.asarray(cube.points)
                      + np.array([0.25, 0.0, 0.0], dtype=np.float32))
    # Every corner's nearest neighbor sits exactly a quarter side away.
    assert tfpv.chamfer_distance(shifted, cube) == 0.25


# -- tubes ----------------------------------------------------------------


def _segment_soup_polydata(points, segments):
    """Line-only PolyData of unordered 2-point segments."""
    result = pv.PolyData()
    result.SetPoints(pv.vtk_points(np.ascontiguousarray(points), deep=True))
    cells = np.hstack([[2, *segment] for segment in segments])
    result.lines = cells
    return result


def test_tube_from_unordered_segments():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    lines = _segment_soup_polydata(points, [[2, 3], [0, 1], [1, 2]])

    result = tfpv.tube(lines, radius=0.25, n_segments=8)
    assert result.n_points == 4 * 8  # one ring per path point
    assert result.n_cells == 3 * 2 * 8  # two triangles per segment and ring

    swept = np.asarray(result.points)
    np.testing.assert_array_equal(np.hypot(swept[:, 1], swept[:, 2]), 0.25)
    assert swept[:, 0].min() == 0.0
    assert swept[:, 0].max() == 3.0


def test_tube_refuses_non_line_input():
    with pytest.raises(ValueError, match="lines only"):
        tfpv.tube(_cube(), radius=0.1)
    with pytest.raises(TypeError):
        tfpv.tube(object(), radius=0.1)


# -- examples ------------------------------------------------------------


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example(name):
    """Import an example module by file; its plotting stays behind main()."""
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_examples_compute():
    base, carved = _example("boolean").compute()
    assert carved.n_cells > 0
    assert carved.trueform.volume() < base.trueform.volume()
    assert set(np.unique(carved.cell_data["trueform_labels"])) == {0, 1}

    torus, gaussian, hills, mean = _example("curvature").compute()
    assert gaussian.shape == (torus.n_points,)
    assert (gaussian > 0).any() and (gaussian < 0).any()
    assert mean.shape == (hills.n_points,)
    assert (mean > 0).any() and (mean < 0).any()

    chunks = _example("csg_fracture").compute()
    assert chunks.n_blocks == 27  # 2 cuts per axis -> 3**3 chunks

    bands, contours = _example("isobands").compute()
    assert bands.n_cells > 0
    assert len(np.unique(bands.cell_data["trueform_labels"])) == 7
    assert contours.GetNumberOfLines() > 0

    source, remeshed, decimated, simplified = \
        _example("remesh_and_simplify").compute()
    assert remeshed.trueform.is_closed()
    assert 0 < decimated.n_cells <= 0.3 * source.n_cells
    assert 0 < simplified.n_cells < source.n_cells
    assert (0.5 * decimated.n_cells <= simplified.n_cells
            <= 2 * decimated.n_cells)

    _, _, _, before, after = _example("alignment").compute()
    assert after < 1e-3 < before

    depth = _example("raycast_depth").compute(resolution=64)
    assert depth.shape == (64, 64)
    assert (~np.isnan(depth)).sum() > 0


# -- packaging -----------------------------------------------------------


def test_entry_point_registers_accessor_without_import():
    code = (
        "import pyvista as pv\n"
        "sphere = pv.Sphere()\n"
        "assert sphere.trueform.is_closed()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
