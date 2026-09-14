# pyvista-trueform caller contract

pyvista-trueform is a boundary and nothing more: PyVista datasets in,
trueform results out. The package computes no geometry, topology, or labels
itself — trueform produces every fact; this package converts, forwards, and
converts back. This document is the caller-facing contract — the AGENTS.md
convention transplanted to runtime, returned by `pyvista_trueform.agents()`
so an agent driving the package in a live session can query the laws and
the surface without leaving it.

`import pyvista_trueform` registers the `.trueform` accessor on
`pyvista.PolyData` and on `pyvista.ImageData`, and the Polydera colormaps
with matplotlib. Installed wheels also reach the accessors through
PyVista's `pyvista.accessors` entry point, so a bare `import pyvista`
serves `.trueform` too.

## The dialect laws

1. **One accessor name, one per carrier.** Mesh operations live on
   `.trueform` of `pyvista.PolyData`, scalar-field operations on
   `.trueform` of `pyvista.ImageData`; module-level functions exist only
   for what is not one dataset's method — conversions, IO, N-ary CSG,
   picking across blocks, registration between two operands, generators,
   colormaps.

2. **Outward, PyVista types.** Geometry answers as a fresh
   `pyvista.PolyData`; a scalar field answers as a fresh
   `pyvista.ImageData`; anything plural answers as a
   `pyvista.MultiBlock`; curves answer as line-only `PolyData`.
   trueform's label arrays ride verbatim as cell data: `trueform_labels`
   (source operand, band, or region) and `trueform_face_labels` (source
   face); a field's samples ride as the `trueform_samples` point-data
   array.

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
   `PolyData`, curve entries take line-only `PolyData`, volume
   conversions take `ImageData` with a single-component point-data array,
   points are `(N, 3)` float32/float64, face indices int32/int64. A
   misspelled keyword fails at this package's signature, not deep in
   trueform.

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
- `volume_to_trueform(dataset, scalars=None)` -> `trueform.Volume` —
  copies the samples; detached by contract, like `to_trueform`. Reads
  the dataset's active point scalars unless `scalars` names another
  array; `dimensions`, `spacing` and `origin` pass through, and the
  direction matrix becomes the volume's 4x4 world pose — a turn about
  the dataset's own origin, so the volume's local space IS the dataset's
  own and no direction can move a coordinate a caller already had. An
  extent that does not start at zero (`extract_subset` keeping its
  coordinates, a `.vti` with a nonzero `WholeExtent`) is folded into
  that origin: a volume's first sample is the first one stored.
- `volume_to_pyvista(volume)` -> `ImageData` — zero-copy; the samples
  ride as the active `trueform_samples` point-data array, and the
  volume's placement (`transformation @ (origin + index * spacing)`)
  goes to PyVista's own `index_to_physical_matrix`, which splits it into
  `origin`, `spacing` and `direction_matrix`. No baking: unlike a Mesh
  transformation, a field's pose is carried by the grid.

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

### Volumes

- `sphere_sdf(dims, spacing, origin, center, radius, *, dtype=None)` ->
  `ImageData` — the analytic sphere field, `distance(point, center) -
  radius` per sample. The only module-level field generator; a mesh's own
  field is `dataset.trueform.signed_distance_field(...)`, and everything
  a field answers lives on `image.trueform`.

### Colormaps

- `polydera_seq()` — the Polydera sequential map, registered with
  matplotlib as `"polydera"` on import.
- `polydera_div()` — the diverging map (orange through mid gray at zero
  to teal), registered as `"polydera_div"`.
- `polydera_cmap(values)` — diverging when `values` cross zero,
  sequential otherwise.

### Introspection

- `agents()` -> str — this document.
- `TrueformAccessor` — the mesh accessor class itself, registered on
  `pyvista.PolyData`.
- `TrueformVolumeAccessor` — the field accessor class, registered on
  `pyvista.ImageData`.
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

### The field a mesh states

- `signed_distance_field(dims, spacing, origin, *, dtype=None,
  mode=None, band=None)` -> `ImageData` — this mesh's signed distance
  field on the stated grid: negative inside, zero on the surface, the
  magnitude the euclidean distance to it. The sign is crossing parity,
  so it wants a CLOSED mesh — nothing refuses an open one, it simply has
  no inside and answers a nonnegative field, which is the silent way to
  get a wrong offset; `outer_shell()` repairs a mesh that should have
  had one. The grid samples this dataset's own coordinates, so build
  `dims` / `spacing` / `origin` from its `bounds`. `mode="banded"` measures only samples
  within `band` voxels of the surface and sweeps the far field — an
  order of magnitude faster, sign exact everywhere, magnitude exact
  inside the band.

## The volume accessor: `image.trueform`

Registered on `pyvista.ImageData` under the same `.trueform` name, with
the same cache contract: the dataset's active point scalars convert into
a `trueform.Volume` once, keyed by the dataset's VTK MTime and the array
read, and every call reuses that instance while both hold. The
conversion COPIES the samples, so the cache is what keeps a 512-cubed
scan from being copied per call. Which array is the field is PyVista's
question: `image.set_active_scalars("ct")` selects one.

The MTime behaves differently here than on a mesh, in the caller's
favour: a VTK data array notifies its dataset, so editing samples
through the dataset (`image.point_data["ct"][k] = ...`,
`image.active_scalars[...] = ...`) advances the MTime and the cache
rebuilds by itself — where the same edit through a `PolyData`'s raw
points would not. What reaches no VTK object is a write through a NumPy
array handed to VTK earlier and still held outside, which is exactly the
buffer `volume_to_pyvista` shares with its volume; call
`image.Modified()` after one.

A volume is a sampled function, so it carries TWO types: `dtype`, what a
sample IS, and `coordinate_dtype`, where the samples STAND. They coincide
for a real-valued field and come apart for an integer one — int16 CT
counts on a float32 millimetre grid, which is how a 512-cubed scan holds
256 MB of samples and not the 512 MB a float32 field would. int16,
uint16 and uint8 are accepted wherever a field is CONSUMED (`isosurface`,
`slice_contours`); a generator emits what it computes, so
`signed_distance_field` and `sphere_sdf` stay real-valued, and
`resampled` refuses an integer field. A sixth dtype is not a sample type
at all: an int64, int32 or bool array is converted to float32 at
construction, and that converted array is what the volume then holds.
Every `dtype=` below names the type that entry EMITS in — float32 or
float64 — and an unstated one is the grid's `coordinate_dtype`, so an
int16 scan surfaces in float32 points. An isovalue is a field value in
the EMITTED type, not the sample type, so a fractional threshold on an
integer field is exactly expressible.

### Conversion

- `to_volume(scalars=None)` -> `trueform.Volume` — the cached volume;
  the same instance while the MTime holds and the same array is asked
  for. The key is the array, not its spelling, so naming the active
  scalars and letting them default share one entry. Treat it as
  read-only.

### Isosurfacing

- `isosurface(iso=None, *, method=None, refine=None, stabilizer=None,
  dtype=None)` -> `PolyData` — the level set as a welded, indexed
  triangle mesh. A corner is inside when `sample < iso`, so a signed
  distance field yields outward windings and a nonzero `iso` on one is
  an offset surface. `method="flying_edges"` (trueform's default) places
  a vertex per crossed grid edge and is defined for any field;
  `"dual_contouring"` places one per surface component of a cell, fitted
  to the field's own crossings, so creases and corners of a
  distance-like field survive and the output is manifold by
  construction. A posed grid answers in world space.
- `slice_contours(plane_origin, u, v, dims2, spacing2, isovalues, *,
  dtype=None)` -> line-only `PolyData` — the field resampled once onto
  an oriented plane's 2D grid and every isovalue contoured on that
  slice, welded into connected polylines. Node `(i, j)` sits at
  `plane_origin + i * spacing2[0] * u + j * spacing2[1] * v`, stated in
  the grid's own space; the result is lifted into world space for a
  posed grid. Contours terminate cleanly at the volume boundary.

### Field CSG

- `union(other, *, dtype=None)` — `min(a, b)`.
- `intersection(other, *, dtype=None)` — `max(a, b)`.
- `difference(other, *, dtype=None)` — `max(a, -b)`, this field minus
  `other`; the other direction is `other.trueform.difference(this)`.

Each answers an `ImageData`; `other` is an `ImageData` (through its own
accessor cache) or a `trueform.Volume`. The combinators are exact on the
zero level set, so the extracted isosurface is exactly the boolean of the
two solids — bound only by the grid's resolution, where the mesh
booleans on `PolyData` are exact everywhere. Matching grids AND matching
poses combine sample-wise and keep that shared pose; anything else
resamples both operands onto a common world-axis-aligned grid and comes
back axis-aligned. Out of an operand's domain is outside its solid.

### Resampling

- `resampled(dims, spacing, origin, *, dtype=None)` -> `ImageData` —
  each target node at `origin + index * spacing` takes the field's
  trilinear value. An axis-aligned grid regrids in its own space,
  clamped at the edge; a posed one regrids THROUGH its pose, the target
  grid being world space, with a sentinel above the field's maximum
  outside the posed domain. The result is always axis-aligned.
  Integer-sampled fields are refused by trueform.

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

6. **Copy vs zero-copy is fixed by direction.** `to_trueform` and
   `volume_to_trueform` copy — detached by contract. `to_pyvista`,
   `curves_to_pyvista` and `volume_to_pyvista` are zero-copy — VTK
   retains the NumPy buffers, so results stay valid after the trueform
   inputs are released. The one exception: baking a `trueform.Mesh`
   transformation into exported points copies them.

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

9. **A volume's placement crosses as a placement.** `spacing`,
   `dimensions` and `origin` pass through in both directions; the
   `direction_matrix` is the volume's 4x4 `transformation` and vice
   versa, as a turn about the dataset's own origin — so the volume's
   local space IS the dataset's own axis-aligned space, a plane or a
   resample grid is stated in the coordinates the dataset already
   speaks, and an undirected grid composes to exactly the identity,
   which trueform reads as unposed. Nothing is baked into samples, which
   is why this stays zero-copy outward even when directed. Three edges:
   the round trip lands in the grid's own `coordinate_dtype` (float32
   for a float32 or integer field), so a float64 placement returns
   rounded; a pose that shears the grid has no such split and PyVista
   refuses it in `index_to_physical_matrix`, naming the shear; and
   trueform keys clamp-versus-sentinel on the pose's presence by exact
   equality, so a direction that merely rounds to the identity is still
   a pose — a resample or boolean will read the far-outside sentinel
   past its box rather than clamping at the edge.

10. **A field is one point-data array, and no transpose.** Every volume
    entry reads the dataset's ACTIVE point scalars —
    `image.set_active_scalars("ct")` chooses, `to_volume(scalars=...)`
    and `volume_to_trueform(dataset, scalars=...)` name one explicitly —
    and every field this package returns rides as `trueform_samples`,
    active. VTK's point order and trueform's voxel order are the same
    flat x-fastest run, so the two carriers meet without a reordering
    pass: `image.point_data["trueform_samples"].reshape(
    image.dimensions, order="F")` IS `volume.samples`, indexed
    `[x, y, z]`. Reshaping in C order transposes the field silently —
    that is the one mistake to make here.

11. **A signed field has a negative inside.** Every field entry reads
    that convention: `isosurface` calls a corner inside when
    `sample < iso`, `signed_distance_field` wants a CLOSED mesh and
    answers a nonnegative field for an open one instead of refusing, and
    the field booleans refuse unsigned (uint8/uint16) operands, naming
    the accepted dtypes. A `0/255`
    segmentation mask is the opposite convention: thresholding it at
    `127.5` extracts the right surface but winds it INTO the
    foreground. Negate the mask into a signed field
    (`127.5 - mask.astype(np.float32)`) for outward windings and for a
    legal boolean operand.

## When you need more

This package binds where trueform produces the fact and PyVista holds the
dataset. trueform's own Python API — expressions, point clouds, primitives,
index maps, everything — is one conversion away: `to_trueform(dataset)`,
`dataset.trueform.to_mesh()`, `volume_to_trueform(image)` or
`image.trueform.to_volume()` inward, `to_pyvista(...)` /
`curves_to_pyvista(...)` / `volume_to_pyvista(...)` outward,
`CsgGraph.native` for a built graph. What PyVista already does well
(plain normals, smoothing, general IO) stays PyVista's; this package
does not shadow it.
