"""
Booleans and picking over two overlapping shapes

A cube and a sphere overlap. The accessor answers their union and
difference with provenance labels riding as cell data; the same pair then
builds one csg_graph whose domains partition space into watertight blocks,
and pick() casts a trueform Ray at the scene to name WHICH block the ray
meets first, and where.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf

from pyvista_trueform import csg_graph, domains, pick


def compute():
    a = pv.Cube().triangulate()
    b = pv.Sphere(radius=0.5, center=(0.5, 0.3, 0.3))
    union = a.trueform.union(b)
    difference = a.trueform.difference(b)
    blocks = domains(csg_graph([a, b]))
    ray = tf.Ray(origin=np.array([3.0, 0.3, 0.3], dtype=np.float32),
                 direction=np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    hit = pick(blocks, ray)
    return union, difference, blocks, hit


def main():
    union, difference, blocks, hit = compute()
    print(f"union: {union.n_cells} cells, difference: {difference.n_cells}")
    print(f"picked block {hit.block_index} of {blocks.n_blocks} "
          f"at {hit.point} (t = {hit.t:.3f})")
    plotter = pv.Plotter(shape=(1, 3))
    plotter.subplot(0, 0)
    plotter.add_mesh(union, scalars="trueform_labels", show_scalar_bar=False)
    plotter.add_text("union")
    plotter.subplot(0, 1)
    plotter.add_mesh(difference, scalars="trueform_labels",
                     show_scalar_bar=False)
    plotter.add_text("difference")
    plotter.subplot(0, 2)
    plotter.add_mesh(blocks, multi_colors=True, opacity=0.6)
    plotter.add_mesh(pv.Sphere(radius=0.04, center=hit.point), color="red")
    plotter.add_text(f"picked block {hit.block_index}")
    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    main()
