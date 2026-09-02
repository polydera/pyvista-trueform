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
from pyvista_trueform import csg_graph

graph = csg_graph([a, b, c])
graph.mesh(tf.op(0) - (tf.op(1) | tf.op(2))).plot()
```

`graph.mesh(...)`, `graph.domains()`, and `graph.intersection_curves()`
answer directly in PyVista types; the rest of the graph's surface
(`created_points`, `forms`, the construction state) lives on
`graph.native`, the underlying `trueform.CsgGraph`.

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

### Picking

`pick` and `closest` answer over any MultiBlock (nested ones flatten;
plain PolyData works too, as block 0) — not just domains — and name WHICH
block:

```python
import trueform as tf
from pyvista_trueform import csg_graph, domains, pick

blocks = domains(csg_graph([a, b]))
hit = pick(blocks, tf.Ray(origin=origin, direction=direction))
# you picked block hit.block_index at hit.point
```

`closest(blocks, query)` does the same by proximity: the query is a point
or a whole dataset/mesh (mesh-to-mesh through the witness-pair entry
`.trueform.closest_point_pair(other)`), and the hit carries the witness
point on the winning block and the euclidean distance.

### Remesh

`mesh.trueform.remeshed(target_length)`, `.decimated(proportion)`, and
`.simplified()` surface trueform's parallel isotropic remesher and
quadric-error tiers — boundary-, feature- and region-aware
(`preserve_regions` labels ride back as cell data).

### Queries

`mesh.trueform.volume()`, `.area()`, `.ray_cast(ray, config=None)`,
`.distance(other)`, `.intersects(other)`, `.closest_point(point)`,
`.closest_point_pair(other)`, `.principal_curvatures()`, `.shape_index()`,
and `.boundary_curves()` answer against the cached mesh, so the spatial
tree amortizes across calls. Queries speak trueform primitives —
`ray_cast` takes a `tf.Ray(origin=..., direction=...)`, single or a whole
batch — and every value this package names "distance" is euclidean.

### Registration

`tfpv.align(source, target, method=...)` fits a 4x4 world-to-world matrix
(`"rigid"`, `"icp"`, `"obb"`, `"knn"`) over any point-bearing datasets or
arrays, ready for `dataset.transform`; `tfpv.chamfer_distance(a, b)` is the
one-way chamfer measure.

### Tubes

`tfpv.tube(lines, radius)` sweeps a triangle tube around line PolyData —
unordered 2-point segments are connected into polylines first — or around
the `(paths, points)` pair any trueform curve producer returns.

### Examples

Six standalone scripts in `examples/`, each with an importable
`compute()` and a plotting `main()` (`python examples/<name>.py`):

- `booleans_and_picking.py` — union and difference with labels, domains,
  and ray picking that names the block.
- `csg_fracture.py` — a sphere minus a grid of cutters through one
  csg_graph expression, read back as volumetric chunks.
- `isobands.py` — height isobands and isocontours overlaid on a surface.
- `remesh_and_simplify.py` — remeshed, decimated, and simplified side by
  side.
- `alignment.py` — a transformed copy registered back with `align`,
  chamfer distance before and after.
- `raycast_depth.py` — an orthographic depth image from one batched
  `tf.Ray` cast.

### Coming with trueform 0.11

`euler_characteristic`, `signed_distance`, orientation verdicts
(`orient_faces_consistently`, `ensure_positive_orientation`), and
`CsgGraph.outer_shell` — the graph-level shell read that spares the second
arrangement build.

## License

Noncommercial use under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE.noncommercial`); commercial licensing via `info@polydera.com` (see
`LICENSE`).
