# pyvista-trueform caller contract

pyvista-trueform is a boundary and nothing more: PyVista datasets in,
trueform results out. The package computes no geometry, topology, or labels
itself — trueform produces every fact; this package converts, forwards, and
converts back. This document is the caller-facing contract — the AGENTS.md
convention transplanted to runtime, returned by `pyvista_trueform.agents()`
so an agent driving the package in a live session can query the laws and
the surface without leaving it.

`import pyvista_trueform` registers the `.trueform` accessor on
`pyvista.PolyData` and the Polydera colormaps with matplotlib. Installed
wheels also reach the accessor through PyVista's `pyvista.accessors` entry
point, so a bare `import pyvista` serves `.trueform` too.

## The dialect laws

1. **One accessor.** Mesh operations live on `.trueform` of
   `pyvista.PolyData`; module-level functions exist only for what is not
   one dataset's method — conversions, IO, N-ary CSG, picking across
   blocks, registration between two operands, generators, colormaps.

2. **Outward, PyVista types.** Geometry answers as a fresh
   `pyvista.PolyData`; anything plural answers as a `pyvista.MultiBlock`;
   curves answer as line-only `PolyData`. trueform's label arrays ride
   verbatim as cell data: `trueform_labels` (source operand, band, or
   region) and `trueform_face_labels` (source face).

3. **Inward, queries speak trueform primitives.** `tf.Point`,
   `tf.Segment`, `tf.Line`, `tf.Plane`, `tf.Triangle`, `tf.AABB`,
   `tf.Ray` — single or batched; a batched primitive answers arrays. A
   faceless dataset (bare-points `PolyData`, `PointSet`) is a batched
   `tf.Point` operand — never a refusal, never a second path. A bare
   `(3,)` array is accepted where a point reads naturally.

4. **Every value named "distance" is euclidean.** trueform's squared
   metrics are converted at the boundary, once. No export of this package
   returns a squared distance under the name "distance".

5. **trueform owns every default.** Every keyword shown `=None` is
   forwarded only when the caller sets it; the omitted ones fall through
   to the trueform callee's own default. The package never restates one.

6. **The boundary is loud.** Types, dtypes, and shapes are validated at
   entry with exact error messages: mesh conversions take polygon-only
   `PolyData`, curve entries take line-only `PolyData`, points are
   `(N, 3)` float32/float64, face indices int32/int64. A misspelled
   keyword fails at this package's signature, not deep in trueform.

## The module surface

### Conversions

- `to_trueform(dataset)` -> `trueform.Mesh` — copies; the mesh is
  detached from the dataset by contract. All-triangle datasets yield
  fixed `(N, 3)` faces, anything else a `trueform.OffsetBlockedArray`;
  indices keep the dataset's VTK storage width.
- `to_pyvista(geometry, *, apply_transformation=True)` -> `PolyData` —
  zero-copy: trueform's offset-block layout IS VTK 9's cell-array
  layout, and VTK retains the NumPy buffers. Takes a `trueform.Mesh` or
  a `(faces, points)` tuple. The one exception to zero-copy: baking a
  Mesh transformation copies the points, by necessity.
- `curves_to_pyvista(paths, points=None)` -> line-only `PolyData` —
  zero-copy; takes the `(paths, points)` pair every trueform curve
  producer returns, as one tuple or two arguments.
- `domains_to_pyvista(cells, ids)` -> `MultiBlock` — block `k` is domain
  `ids[k]`, named `str(ids[k])`; each block converts zero-copy.

### IO

- `read(path, *, index_dtype=None, ngon=None, dtype=None)` ->
  `PolyData` — dispatched on suffix: `.stl` (parallel, duplicate
  vertices welded), `.obj` (polygon sizes preserved); `ngon`/`dtype` are
  `.obj`-only.
- `write(path, dataset, *, transformation=None)` — dispatched on suffix:
  `.stl` (triangles only), `.obj` (any polygon sizes). Raises `OSError`
  when trueform fails to write.

### N-ary CSG and arrangements

- `csg_graph(datasets, *, sheets=None, mode=None, tolerance=None,
  resolve_crossings=None, within=None, triangulation=None)` ->
  `CsgGraph` — one arrangement of N operands (PolyData or
  `trueform.Mesh`, each through its own accessor cache), arbitrarily
  many boolean expressions answered against it. A single operand is
  legal: its own self arrangement.
- `mesh_arrangements(datasets, *, return_curves=False, mode=None,
  tolerance=None, resolve_crossings=None, resolve_self_crossings=None,
  within=None, triangulation=None)` -> labeled `PolyData` — every face
  split along every intersection curve, provenance as `trueform_labels`
  / `trueform_face_labels`; with `return_curves=True` also the curves as
  a second, line-only `PolyData`.
- `domains(datasets_or_graph, expr=None, *, selection=None,
  exclude_outer_shell=None, ignore_open_fragments=None,
  return_source_ids=None, return_index_map=None)` -> `MultiBlock` —
  every kept volumetric domain as a named block. Takes a sequence of
  datasets (the graph is built here) or a prebuilt `CsgGraph` /
  `trueform.CsgGraph`.
- `split_into_domains(arranged, *, ignore_open_fragments=None,
  exclude_outer_shell=None)` -> `MultiBlock` — splits an
  already-arranged mesh (a `mesh_arrangements` output, a
  `CsgGraph.mesh` read) through `trueform.domain_labels`; no arrangement
  is built. A mesh that still self-intersects goes through
  `dataset.trueform.domains()` instead.

### Picking across blocks

- `pick(target, ray, *, config=None)` -> `RayHit | None` — the first
  face of the target in the ray's way, naming its block. `target` is a
  `PolyData` (its own single block, index 0) or a `MultiBlock` (nested
  ones flatten depth-first, `None` blocks are skipped and keep their
  numbers); `ray` is a single `tf.Ray` (a batch is refused); `config` is
  the `(min_t, max_t)` parametric range.
- `closest(target, query, *, radius=None)` -> `ClosestHit | None` — the
  block nearest to `query` with its witness. `query` is a `(3,)` point,
  a dataset, or a `trueform.Mesh`; a block with nothing within `radius`
  is skipped.
- `RayHit(block_index, block, face, point, t)` and
  `ClosestHit(block_index, block, face, point, distance)` — NamedTuples;
  `point` is the hit / witness point on the winning block, `distance` is
  euclidean.

### Registration

Each takes `source` and `target` as any PyVista dataset with `.points`,
a bare `(N, 3)` array, or a `trueform.PointCloud`, sharing one dtype.
Every `align_*` returns the DELTA: a homogeneous `(4, 4)` matrix mapping
the source's CURRENT points onto the target, nothing of the source's own
transformation history composed in — apply it to the source itself,
`source.transform(matrix, inplace=False)`.

- `align_rigid(source, target)` — Kabsch; point-to-point correspondence
  required.
- `align_similarity(source, target)` — rotation + uniform scale +
  translation; same correspondence requirement.
- `align_icp(source, target, *, max_iterations=None, n_samples=None,
  k=None, sigma=None, outlier_proportion=None,
  min_relative_improvement=None, ema_alpha=None)` — iterative closest
  point; no correspondence.
- `align_obb(source, target, *, sample_size=None)` — oriented-bounding-
  box alignment; no correspondence.
- `align_knn(source, target, *, k=None, sigma=None,
  outlier_proportion=None)` — one soft k-nearest-neighbor step; no
  correspondence.
- `chamfer_distance(source, target)` -> float — one-way mean
  nearest-neighbor distance; average the two directions for the
  symmetric measure.

### Lines and tubes

- `connect_lines(dataset)` -> line-only `PolyData` — unordered 2-point
  segments assembled into polylines over the dataset's own point ids;
  cells that already are polylines pass through. The result is detached.
- `tube(lines, radius, *, n_segments=None)` -> `PolyData` — a triangle
  tube around every polyline of a line-only `PolyData` or a
  `(paths, points)` pair; closed loops are auto-detected.

### Generators

All return a fresh `PolyData` with outward-facing normals (CCW winding);
`dtype`/`index_dtype` set point and face-index dtypes (trueform defaults:
float32, int32).

- `box(width, height, depth, *, width_ticks=None, height_ticks=None,
  depth_ticks=None, dtype=None, index_dtype=None)`
- `sphere(radius, *, stacks=None, segments=None, dtype=None,
  index_dtype=None)` — UV sphere.
- `cylinder(radius, height, *, segments=None, dtype=None,
  index_dtype=None)` — capped, centered along z.
- `plane(width, height, *, width_ticks=None, height_ticks=None,
  dtype=None, index_dtype=None)` — XY plane, normal +z.

### Colormaps

- `polydera_seq()` — the Polydera sequential map, registered with
  matplotlib as `"polydera"` on import.
- `polydera_div()` — the diverging map (orange through mid gray at zero
  to teal), registered as `"polydera_div"`.
- `polydera_cmap(values)` — diverging when `values` cross zero,
  sequential otherwise.

### Introspection

- `agents()` -> str — this document.
- `TrueformAccessor` — the accessor class itself, registered on
  `pyvista.PolyData`.
- `__version__` — the installed distribution version.

## The accessor: `dataset.trueform`

Every method answers against the cached `trueform.Mesh` (see the cache
contract below), so the spatial tree, face membership, and manifold edge
link amortize across calls. Operands named `other` are a `PolyData`
(through its own accessor cache) or a `trueform.Mesh`.

### Conversion

- `to_mesh()` -> `trueform.Mesh` — the cached mesh; the same instance
  while the dataset's MTime holds. Treat it as read-only.

### Booleans

- `union(other, *, return_curves=False, sheets=None)`
- `intersection(other, *, return_curves=False, sheets=None)`
- `difference(other, *, return_curves=False, sheets=None)` — this mesh
  minus `other`; the other direction is `other.trueform.difference(this)`.

Each returns a labeled `PolyData` (`trueform_labels` = source operand 0/1,
`trueform_face_labels` = source face); with `return_curves=True` also the
intersection curves as a second, line-only `PolyData`. `sheets` names
operand indices (0/1) declared as oriented separators that bound no
volume.

### Intersection curves

- `intersection_curves(other, *, mode=None, tolerance=None,
  resolve_crossings=None, resolve_self_crossings=None)` -> line-only
  `PolyData`.
- `self_intersection_curves(*, mode=None, tolerance=None,
  resolve_crossings=None, resolve_self_crossings=None)` -> line-only
  `PolyData` — trueform defaults both crossing options to True here.

### Scalar-field cuts

- `isocontours(scalars, threshold)` -> line-only `PolyData` — `scalars`
  is a `point_data` array name or an array; `threshold` a value or an
  array of values.
- `isobands(scalars, cut_values, *, selected_bands=None,
  return_curves=False)` -> labeled `PolyData` — the mesh recut into
  bands; the band rides as `trueform_labels`, the source face as
  `trueform_face_labels`.

### Repair and processing

- `domains(expr=None, *, selection=None, exclude_outer_shell=None,
  ignore_open_fragments=None, return_source_ids=None,
  return_index_map=None)` -> `MultiBlock` — this mesh's
  self-decomposition into volumetric domains, through its own one-operand
  `trueform.CsgGraph`.
- `polygon_arrangements(*, return_curves=False, mode=None,
  tolerance=None, resolve_crossings=None, resolve_self_crossings=None,
  triangulation=None)` -> labeled `PolyData` — the mesh split at its own
  self-intersection curves; provenance as `trueform_face_labels`.
- `outer_shell()` -> `PolyData` — repair to the boundary of the union of
  everything the mesh encloses, free of self-intersections.
- `cleaned(tolerance=None, *, return_index_map=None,
  remove_duplicate_primitives=None, remove_unreferenced_points=None)` ->
  `PolyData` — duplicate vertices and degenerate faces removed;
  `tolerance` merges vertices within that world-coordinate distance.
  With `return_index_map=True` also the face and point index maps,
  passed through untouched.
- `triangulated()` -> `PolyData` — every face triangulated on its own
  boundary, shared edges one identity in both faces.

### Remeshing

Each returns a `PolyData`; with `preserve_regions=` (one label per input
face) the surviving labels ride as `trueform_labels`.

- `remeshed(target_length, *, iterations=None, relaxation_iters=None,
  min_quality=None, lambda_=None, preserve_boundary=None,
  use_quadric=None, parallel=None, feature_angle=None,
  feature_weight=None, preserve_regions=None)` — isotropic remesh
  toward `target_length` edges.
- `decimated(target_proportion, *, min_quality=None,
  preserve_boundary=None, stabilizer=None, parallel=None,
  feature_angle=None, feature_weight=None, preserve_regions=None)` —
  quadric-error decimation to a face-count proportion.
- `simplified(*, error_rel=None, optimize_iterations=None,
  iterations=None, relaxation_iters=None, lambda_=None,
  min_quality=None, preserve_boundary=None, stabilizer=None,
  parallel=None, feature_angle=None, feature_weight=None,
  preserve_regions=None)` — quadric-error simplification to an error
  budget; no target face count.

### Topology reads

- `connected_components(*, expected_number_of_components=None)` ->
  `(n, labels)` — component count and a per-face int32 label array over
  manifold-edge adjacency.
- `split_components()` -> `MultiBlock` — one block per component, named
  `str(k)`, points reindexed to the ones the component uses.
- `non_manifold_edges()` / `boundary_edges()` -> line-only `PolyData` —
  one line cell per edge, ids naming this dataset's own points (the full
  point array rides along, so ids read straight back); empty `PolyData`
  when there are none.
- `non_manifold_paths()` / `boundary_paths()` -> line-only `PolyData` —
  the same edges assembled into polylines/loops, same point ids.
- `boundary_curves()` -> line-only `PolyData` — the boundary loops over
  a compacted point set of their own instead.

### Diagnostics and measures

- `is_closed()`, `is_open()`, `is_manifold()`, `is_non_manifold()` ->
  bool.
- `area()`, `volume()`, `signed_volume()`, `mean_edge_length()` ->
  float.
- `euler_characteristic()` -> int — `V - E + F`, each undirected edge
  counted once.

### Spatial queries

- `ray_cast(ray, config=None)` — `ray` is a `tf.Ray`, single or batch;
  `config` the `(min_t, max_t)` range. Single: `(face_id, t)` or `None`;
  batch: `(face_ids, ts)` arrays, a miss `-1` / `NaN`.
- `distance(other)` -> float — euclidean distance to a dataset, mesh,
  trueform primitive (a batched one answers a `(N,)` array), or `(3,)`
  point.
- `signed_distance(other)` -> `(N,)` array — from every point of THIS
  dataset to `other`'s surface, negative inside; this dataset's points
  go as one batched `tf.Point`, so a faceless dataset queries just as
  well.
- `intersects(other)` -> bool — dataset, mesh, or primitive (a batched
  one answers a 0/1 `(N,)` array).
- `closest_point(query_point, *, radius=None)` ->
  `(face_id, distance, point)` or `None` when `radius` bounds the search
  and nothing lies within; a batched primitive answers
  `(face_ids, distances, points)` arrays, a miss `-1`.
- `closest_points(query, k, *, radius=None)` -> list of up to `k`
  `(face_id, distance, point)` tuples, closest first; a batched
  primitive answers `(face_ids, distances, points, counts)` arrays.
- `closest_point_pair(other, *, radius=None)` ->
  `((face_id, other_id), (distance, point, other_point))` or `None` — a
  faced operand answers mesh-to-mesh (`other_id` names its face), a
  faceless one queries its points as one batched `tf.Point`
  (`other_id` names its point).
- `principal_curvatures(*, k=None, directions=None)` — per-vertex
  `(k0, k1)`; with `directions=True` also `(d0, d1)`.
- `shape_index(*, k=None)` — per-vertex shape index in `[-1, 1]`.

## The CsgGraph wrapper

Built by `csg_graph`. Holds the native graph and nothing else; the
readers forward and convert.

- `mesh(expr=None, *, selection=None, inside=None,
  return_source_ids=None, return_index_map=None)` -> `PolyData` — the
  boolean result of `expr` (`tf.op(i)` combined with `|`, `&`, `-`,
  `~`); with no expression, the full arrangement mesh. With
  `return_source_ids=True` provenance rides as cell data; with
  `return_index_map=True` returns `(polydata, index_map)`, the
  `trueform.MeshIndexMap` untouched.
- `domains(expr=None, *, selection=None, exclude_outer_shell=None,
  ignore_open_fragments=None, return_source_ids=None,
  return_index_map=None)` -> `MultiBlock` — every kept volumetric
  domain, block `k` named `str(ids[k])`. `return_source_ids=True` adds
  two `trueform.OffsetBlockedArray` of per-cell provenance, untouched;
  `return_index_map=True` adds the `trueform.DomainsIndexMap`,
  untouched.
- `intersection_curves()` -> line-only `PolyData` — the cross-operand
  seams (coincident walls excluded).
- `outer_shell()` -> `PolyData` — the boundary between the unbounded
  universe and everything the operands enclose; a structural read off
  the graph already built.
- `native` — the underlying `trueform.CsgGraph`: everything the wrapper
  does not convert (`created_points`, `forms`, `sheets`, construction
  state) lives here, in trueform's own types.

## The contracts that bite

1. **The cache is whole-value, keyed by MTime.** The accessor holds one
   `trueform.Mesh` per dataset, keyed by the dataset's VTK modification
   time — one integer compare per access. While the MTime holds, every
   call reuses the same instance; when it changes, the mesh is discarded
   whole and rebuilt. VTK only advances the MTime through its own API:
   mutating a raw NumPy view (`np.asarray(pd.points)[0] = ...`) does NOT
   bump it, and the accessor keeps serving the stale mesh — call
   `pd.Modified()` after such edits. Assignments through PyVista's own
   surface (`pd.points = ...`, `pd.points[0] = ...`) notify VTK already.

2. **`align_*` returns the delta.** The `(4, 4)` matrix maps the
   source's current points onto the target; nothing of the source's own
   transformation history is composed in. Apply it to the source itself:
   `source.transform(matrix, inplace=False)`.

3. **The universe block.** The graph readers (`CsgGraph.domains`,
   `dataset.trueform.domains`, module-level `domains`) exclude the
   unbounded universe by trueform's graph default. `split_into_domains`
   reads through `trueform.domain_labels`, whose default KEEPS the
   universe as a domain — pass `exclude_outer_shell=True` there to drop
   it.

4. **Sheets bound no volume.** An operand named in `sheets` (indices
   into the operand list) is an oriented separator: it cuts through the
   boolean algebra without enclosing a volume. Available on the accessor
   booleans (`sheets={0}` or `{1}`) and on `csg_graph`.

5. **Non-triangle operands normalize.** `csg_graph` and
   `mesh_arrangements` keep a triangle graph when every operand is
   all-triangle; otherwise every operand is re-expressed as dynamic
   (variable-sized) faces first — lossless. Operands with differing face
   index dtypes widen to int64.

6. **Copy vs zero-copy is fixed by direction.** `to_trueform` copies —
   detached by contract. `to_pyvista` and `curves_to_pyvista` are
   zero-copy — VTK retains the NumPy buffers, so results stay valid
   after the trueform inputs are released. The one exception: baking a
   `trueform.Mesh` transformation into exported points copies them.

7. **The escape hatches.** `dataset.trueform.to_mesh()` and
   `to_trueform` cross into trueform's own Python API; `to_pyvista`
   crosses back; `CsgGraph.native` is the raw graph. Everything trueform
   offers beyond this surface stays reachable by composition — nothing
   is walled off.

8. **The colormaps register on import.** `"polydera"` (sequential) and
   `"polydera_div"` (diverging) resolve anywhere matplotlib accepts a
   colormap name, e.g. `dataset.plot(cmap="polydera_div")`;
   `polydera_cmap(values)` picks between them by whether the values
   cross zero.

## When you need more

This package binds where trueform produces the fact and PyVista holds the
dataset. trueform's own Python API — expressions, point clouds, primitives,
index maps, everything — is one conversion away: `to_trueform(dataset)` or
`dataset.trueform.to_mesh()` inward, `to_pyvista(...)` /
`curves_to_pyvista(...)` outward, `CsgGraph.native` for a built graph.
What PyVista already does well (plain normals, smoothing, general IO)
stays PyVista's; this package does not shadow it.
