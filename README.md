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

## License

Noncommercial use under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE.noncommercial`); commercial licensing via `info@polydera.com` (see
`LICENSE`).
