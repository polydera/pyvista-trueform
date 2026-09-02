# pyvista-trueform

[trueform](https://trueform.polydera.com) operations on
[PyVista](https://pyvista.org) meshes: exact booleans, intersection curves,
self-intersection repair, isocontours and isobands, N-ary CSG — through one
`.trueform` accessor on `pyvista.PolyData`.

## Install

```
pip install pyvista-trueform
```

## Use

```python
import pyvista as pv
import pyvista_trueform  # registers the accessor

a = pv.Cube().triangulate()
b = pv.Cube(center=(0.5, 0.5, 0.5)).triangulate()
a.trueform.union(b).plot()
```

Every operation returns a fresh `pv.PolyData`; trueform's label arrays ride
along as cell data (`trueform_labels`, `trueform_face_labels`). The accessor
converts the dataset into a `trueform.Mesh` once and caches it keyed by the
dataset's VTK modification time, so trueform's lazily built structures
(spatial tree, face membership, edge link) amortize across calls. When the
dataset changes, the next call rebuilds the mesh from scratch.

One VTK gotcha: mutating a raw NumPy view of the underlying arrays does not
advance the modification time, so the accessor would keep serving the stale
mesh. Call `polydata.Modified()` after such edits; assignments through
PyVista's own surface notify VTK already.

### N-ary CSG

Build one arrangement of N operands and answer arbitrarily many boolean
expressions against it:

```python
import trueform as tf
from pyvista_trueform import csg_graph, to_pyvista

graph = csg_graph([a, b, c])
to_pyvista(graph.mesh(tf.op(0) - (tf.op(1) | tf.op(2)))).plot()
```

### Conversions

`to_trueform(polydata)` copies polygon geometry into a detached
`trueform.Mesh`; `to_pyvista(mesh_or_faces_points)` and
`curves_to_pyvista(paths, points)` go the other way zero-copy — trueform's
offset-block faces are exactly VTK 9's cell-array layout, and VTK retains
the NumPy buffers. Everything trueform offers beyond the accessor's headline
methods stays reachable by composition through these functions.

### IO

`tfpv.read(path)` and `tfpv.write(path, dataset)` move meshes through
trueform's parallel STL and OBJ readers and writers, dispatched on the path
suffix; reading converts zero-copy, STL welds duplicate vertices on the way
in.

### Domains and N-ary arrangements

`tfpv.domains([a, b, c])` partitions space by every surface and returns each
watertight volumetric domain as a block of a `pv.MultiBlock`, named by its
domain id — pass a prebuilt `csg_graph(...)` and an expression to restrict
it. `tfpv.mesh_arrangements([a, b, c])` returns the whole arrangement as one
labeled PolyData instead.

### Remesh

`mesh.trueform.remeshed(target_length)`, `.decimated(proportion)`, and
`.simplified()` surface trueform's parallel isotropic remesher and
quadric-error tiers — boundary-, feature- and region-aware
(`preserve_regions` labels ride back as cell data).

### Queries

`mesh.trueform.volume()`, `.area()`, `.ray_cast(origin, direction)`,
`.distance(other)`, `.intersects(other)`, `.closest_point(point)`,
`.principal_curvatures()`, `.shape_index()`, and `.boundary_curves()` answer
against the cached mesh, so the spatial tree amortizes across calls.

### Registration

`tfpv.align(source, target, method=...)` fits a 4x4 world-to-world matrix
(`"rigid"`, `"icp"`, `"obb"`, `"knn"`) over any point-bearing datasets or
arrays, ready for `dataset.transform`; `tfpv.chamfer_distance(a, b)` is the
one-way chamfer measure.

### Tubes

`tfpv.tube(lines, radius)` sweeps a triangle tube around line PolyData —
unordered 2-point segments are connected into polylines first — or around
the `(paths, points)` pair any trueform curve producer returns.

### Coming with trueform 0.11

`euler_characteristic`, `signed_distance`, orientation verdicts
(`orient_faces_consistently`, `ensure_positive_orientation`), and
`CsgGraph.outer_shell` — the graph-level shell read that spares the second
arrangement build.

## License

Noncommercial use under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE.noncommercial`); commercial licensing via `info@polydera.com` (see
`LICENSE`).
