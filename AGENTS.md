# pyvista-trueform agent contract

This package is a boundary and nothing more: PyVista datasets in, trueform
results out, ~1000 lines total. trueform owns every computation; PyVista owns
every dataset. Work here is judged by whether the boundary stays this thin.

Developed jointly by Žiga Sajovic and Claude.

## The laws

1. **The cache is whole-value.** The accessor caches one `trueform.Mesh`
   per dataset, keyed by the dataset's VTK MTime — one integer compare per
   access. While the MTime holds, every call reuses the same instance, so
   trueform's lazily built structures (tree, face membership, edge link)
   amortize. When it changes, the mesh is discarded whole and rebuilt.
   Never partial refresh, never per-array owners, never storage tracking —
   that design was considered and rejected as overengineering.

2. **Conversion direction decides copying.** `to_trueform` copies — the
   mesh is detached from the dataset by contract. `to_pyvista` and
   `curves_to_pyvista` are zero-copy: trueform's offset-block layout IS
   VTK 9's cell-array layout, and VTK retains the NumPy buffers. The one
   exception is a baked `trueform.Mesh` transformation, which must copy
   points; nothing else may quietly copy or quietly alias.

3. **Nothing is rederived.** This package never computes geometry,
   topology, or labels — it converts, forwards to trueform's public Python
   API, and converts back. Labels ride verbatim as cell data. If an
   operation needs a fact, trueform produces it.

4. **The boundary is loud.** Types, dtypes, and shapes are validated at
   entry with exact error messages. No silent coercion, and no silent
   feature loss: a dependency floor is a feature floor (the pyvista floor
   is the version whose accessor registry this package registers into).

5. **The package speaks two dialects, one per direction.** Inward,
   queries speak trueform primitives: `ray_cast` and `pick` take a
   `tf.Ray` (single or batch), a point query is an array or a batched
   `tf.Point` — a points-only PyVista dataset wraps its `.points` as one
   batched primitive, never a refusal, never a second path. Outward,
   readers answer in PyVista types: `PolyData` with labels as cell data,
   `MultiBlock` for anything plural, line PolyData for curves — the
   `CsgGraph` wrapper exists for exactly this, with `.native` as the one
   escape hatch. And every value this package names "distance" is
   euclidean; a squared metric is converted at the boundary, once.

6. **Every claim is a fixture.** Cache identity (`is`), the raw-mutation
   MTime gotcha, zero-copy owner-safety under `gc`, entry-point
   registration in a clean subprocess — each documented behavior has the
   test that proves it, in `tests/test_pyvista_trueform.py`.

## Boundaries with the neighbors

- trueform's Python surface is the only trueform this package sees. Never
  reach into its internals; its own contract lives in the trueform
  repository (`agents/python_layer.md`).
- PyVista is reached through its public API plus the VTK cell-array/
  numpy_support seam used by the conversions. Accessor registration goes
  through `pyvista.register_dataset_accessor` and the `pyvista.accessors`
  entry point — both, so `import pyvista_trueform` and a bare
  `import pyvista` reach the same accessor.

## Validation

```bash
python -m pytest tests
```
