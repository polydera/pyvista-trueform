![pyvista-trueform](https://raw.githubusercontent.com/polydera/pyvista-trueform/main/assets/header.png)

![Six overlapping spheres split into their domains by one csg_graph arrangement](https://raw.githubusercontent.com/polydera/pyvista-trueform/main/assets/hero_domains_multiblock.png)

Exact mesh booleans, intersection curves, self-intersection repair,
isocontours and isobands, and N-ary CSG on [PyVista](https://pyvista.org)
meshes. PyVista in, PyVista out, through one `.trueform` accessor on
`pyvista.PolyData`.

## Getting Started

```bash
pip install pyvista-trueform
```

**The accessor** registers itself on import and answers through
`.trueform` on any `pyvista.PolyData`:

```python
import pyvista as pv
import pyvista_trueform  # registers the accessor

pv.Cube().trueform.volume()
```

**The cache** is whole-value, keyed by the dataset's VTK modification
time — the held instance is what lets trueform's spatial tree, face
membership, and edge link build lazily once:

```python
import numpy as np

a = pv.Cube()
mesh = a.trueform.to_mesh()
assert a.trueform.to_mesh() is mesh  # same cached instance, MTime unchanged

np.asarray(a.points)[0] += 1.0       # raw mutation: MTime NOT bumped
a.Modified()                         # tell VTK, so the cache rebuilds
assert a.trueform.to_mesh() is not mesh
```

**A worked example** — a boolean carves the shape, `domains` reads the
pieces the arrangement made, and `pick` names the one a ray struck:

```python
import numpy as np
import pyvista as pv
import pyvista_trueform as tfpv
import trueform as tf

a = pv.Cube()
b = pv.Cube(center=(0.5, 0.5, 0.5))

carved = a.trueform.difference(b)
carved.cell_data["trueform_labels"]  # per-face source: 0 (a) or 1 (b)
blocks = tfpv.domains([a, b])        # every overlap pocket, its own block
ray = tf.Ray(origin=np.array([-2, 0, 0], dtype=np.float32),
            direction=np.array([1, 0, 0], dtype=np.float32))
hit = tfpv.pick(blocks, ray)
hit.block_index                      # which block the ray struck
```

**The contract** the package holds itself to — dialect laws, the full
surface, the gotchas — is one call away:

```python
print(tfpv.agents())  # the package contract, for agents and humans alike
```

The sections below cover the rest of the surface these three lean on.

### N-ary CSG

Build one arrangement of N operands and answer arbitrarily many boolean
expressions against it:

```python
from pyvista_trueform import csg_graph

c = pv.Cube(center=(0.25, 0.5, 0.0))  # quads, mixed ngons: normalized losslessly
graph = csg_graph([a, b, c])
graph.mesh(tf.op(0) - (tf.op(1) | tf.op(2)))

graph.domains()                       # every watertight piece, as a MultiBlock
graph.intersection_curves()           # the seams, as line PolyData
graph.outer_shell()                   # boundary of everything the operands enclose
graph.native                          # escape hatch: the raw trueform.CsgGraph

csg_graph([a])                        # one operand is legal: its own self arrangement
```

### Conversions

`to_trueform` and `to_pyvista` cross the boundary; only `to_trueform`
copies:

```python
from pyvista_trueform import to_trueform, to_pyvista, curves_to_pyvista

mesh = to_trueform(a)                        # detached trueform.Mesh, copies the geometry
to_pyvista(mesh)                             # zero-copy: shares mesh's own NumPy buffers
curves_to_pyvista(tf.boundary_curves(mesh))  # any (paths, points) pair, zero-copy
```

### IO

```python
from pyvista_trueform import read, write

write("cube.obj", a)                          # dispatched on suffix; .obj keeps a's own quad faces
read("cube.obj")                              # zero-copy conversion back to PolyData

write("cube.stl", a.trueform.triangulated())  # STL: triangles only
read("cube.stl")                              # STL welds duplicate vertices on the way in
```

### Domains and N-ary arrangements

```python
from pyvista_trueform import domains, mesh_arrangements, split_into_domains

domains([a, b, c])            # every watertight domain, named by id, as a MultiBlock
mesh_arrangements([a, b, c])  # the whole arrangement instead, as one labeled PolyData
a.trueform.domains()          # one mesh's own overlap pockets, from its self arrangement

arranged = mesh_arrangements([a, b, c])
split_into_domains(arranged, exclude_outer_shell=True)  # already cut: labeled and
                                                        # split, no second arrangement
```

### Picking

`pick` and `closest` answer over any MultiBlock (nested ones flatten; a
plain PolyData works too, as block 0):

```python
from pyvista_trueform import pick, closest

blocks = domains(csg_graph([a, b]))
ray = tf.Ray(origin=np.array([-2, 0, 0], dtype=np.float32),
            direction=np.array([1, 0, 0], dtype=np.float32))
pick(blocks, ray)                 # hit.block_index, hit.point: first block struck
closest(blocks, [0.0, 0.0, 0.0])  # same shape, by proximity instead of a ray
```

### Remesh

```python
tri = a.trueform.triangulated()

tri.trueform.remeshed(0.1)             # isotropic, target edge length
tri.trueform.decimated(0.5)            # quadric-error, target face proportion
tri.trueform.simplified()              # quadric-error, to an error budget instead

labels = np.zeros(tri.n_cells, dtype=np.int32)
remeshed = tri.trueform.remeshed(0.1, preserve_regions=labels)
remeshed.cell_data["trueform_labels"]  # preserve_regions rides back as cell data
```

### Queries

```python
a.trueform.volume()                  # .area(), .signed_volume(), .mean_edge_length() too
a.trueform.is_closed()               # .is_open(), .is_manifold(), .is_non_manifold(),
                                     # .euler_characteristic() too
a.trueform.distance(b)               # euclidean; .intersects(b) too
a.trueform.signed_distance(b)        # negative inside b; batched over a's own points
a.plot(scalars=a.trueform.signed_distance(b), cmap="polydera_div")  # packaged map, registered on import
a.trueform.closest_point([0, 0, 0])  # (face_id, distance, point)
a.trueform.closest_points([2, 0, 0], k=3)  # the k nearest, closest first
a.trueform.closest_point_pair(b)     # witness pair between a and b
a.trueform.principal_curvatures()    # (k0, k1); .shape_index() too

seg = tf.Segment(np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float32))
a.trueform.distance(seg)             # queries take trueform primitives; .closest_point(seg) too

n, labels = a.trueform.connected_components()  # (n, per-face component labels)
a.trueform.split_components()        # one block per component, as a MultiBlock
a.trueform.boundary_curves()         # open edges, as line PolyData
a.trueform.boundary_edges()          # the dataset's own point ids instead; .boundary_paths(),
                                     # .non_manifold_edges(), .non_manifold_paths() too

ray = tf.Ray(origin=np.array([-2, 0, 0], dtype=np.float32),
            direction=np.array([1, 0, 0], dtype=np.float32))
a.trueform.ray_cast(ray)             # (face_id, t); config=(min_t, max_t) bounds it
```

All of these answer against the cached mesh, so the spatial tree
amortizes across calls.

### Registration

```python
from pyvista_trueform import (align_rigid, align_similarity, align_icp,
                             align_obb, align_knn, chamfer_distance)

align_rigid(a, b)       # Kabsch: rotation + translation, correspondence required
align_similarity(a, b)  # + uniform scale, same correspondence requirement
align_icp(a, b)         # iterative closest point, no correspondence needed
align_obb(a, b)         # oriented-bounding-box alignment, no correspondence
align_knn(a, b)         # one soft-correspondence step
chamfer_distance(a, b)  # one-way chamfer measure

matrix = align_rigid(a, b)                    # 4x4 delta: maps a's current points onto b
aligned = a.transform(matrix, inplace=False)  # apply to a itself to align it with b
```

### Lines and tubes

```python
from pyvista_trueform import connect_lines, tube

line = pv.Line((0, 0, 0), (1, 0, 0))
tube(line, radius=0.1)  # triangle tube; unordered 2-point segments connect first
connect_lines(line)     # just the assembly: segments in, polylines out, same point ids
```

### Generators

```python
tfpv.box(2.0, 1.0, 3.0)                                   # ticks subdivide each axis
tfpv.sphere(1.0, stacks=20, segments=20)
tfpv.cylinder(1.0, 2.0, segments=20)
tfpv.plane(10.0, 5.0)                                     # ticks subdivide each axis
tfpv.sphere(1.0, dtype=np.float64, index_dtype=np.int64)  # trueform's defaults apply when omitted
```

### Examples

![A sphere fractured by a grid of cutters in one csg_graph expression, read back as exploded domain chunks](https://raw.githubusercontent.com/polydera/pyvista-trueform/main/assets/hero_csg_fracture.png)

Seven standalone scripts in `examples/`, each with an importable
`compute()` and a plotting `main()` (`python examples/<name>.py`); the
gallery ships in the Polydera color scheme through the shared
`examples/_theme.py`:

- `boolean.py` — a boolean difference recut live under a dragged sphere
  widget, the result labeled by source.
- `csg_fracture.py` — a sphere minus a grid of cutters through one
  csg_graph expression, read back as volumetric chunks; clicking a chunk
  names it through `closest`.
- `isobands.py` — height isobands and isocontours overlaid on a
  surface, recut live from a band-count slider.
- `signed_distance.py` — a hills surface recolored live by signed
  distance to a dragged torus probe, diverging about the zero seam.
- `curvature.py` — Gaussian curvature across a torus and mean curvature
  across a hills surface, from one `principal_curvatures` call each.
- `remesh_and_simplify.py` — remeshed, decimated, and simplified side by
  side.
- `raycast_depth.py` — an orthographic depth image from one batched
  `tf.Ray` cast.

### Coming with trueform 0.11

Orientation verdicts (`orient_faces_consistently`,
`ensure_positive_orientation`) — the Python binding still returns only the
repaired faces array, no verdict yet.

## License

Dual-licensed:
- **Noncommercial**: [PolyForm Noncommercial License 1.0.0](./LICENSE.noncommercial)
- **Commercial**: Contact [info@polydera.com](mailto:info@polydera.com)
