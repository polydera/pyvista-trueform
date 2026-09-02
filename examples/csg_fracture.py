"""
Cellular fracture from one csg_graph expression

A sphere minus a grid of thin axis-aligned slabs, carved in a single
csg_graph expression — one arrangement of all eight operands, built once.
The domains inside the expression's region are the resulting chunks, read
back as a MultiBlock and plotted exploded in distinct colors. Clicking a
chunk asks closest() to name it, and the named chunk lights up.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import operator
from functools import reduce

import numpy as np
import pyvista as pv
import trueform as tf

from pyvista_trueform import closest, csg_graph, domains


def _slab(axis, offset):
    lengths = [2.4, 2.4, 2.4]
    lengths[axis] = 0.02
    center = [0.0, 0.0, 0.0]
    center[axis] = offset
    return pv.Cube(center=center, x_length=lengths[0], y_length=lengths[1],
                   z_length=lengths[2]).triangulate()


def compute(cuts_per_axis=2):
    solid = pv.Sphere(radius=1.0, theta_resolution=48, phi_resolution=48)
    offsets = np.linspace(-1.0, 1.0, cuts_per_axis + 2)[1:-1]
    cutters = [_slab(axis, offset)
               for axis in range(3) for offset in offsets]
    graph = csg_graph([solid, *cutters])
    carve = reduce(operator.or_,
                   (tf.op(k) for k in range(1, len(cutters) + 1)))
    return domains(graph, tf.op(0) - carve)


def main():
    import _theme
    chunks = compute()
    print(f"{chunks.n_blocks} chunks")
    exploded = pv.MultiBlock()
    for block in chunks:
        shifted = block.copy()
        shifted.points = shifted.points + 0.35 * np.asarray(shifted.center)
        exploded.append(shifted)
    plotter = pv.Plotter(theme=_theme.theme())

    def draw_chunks(selected):
        for k, block in enumerate(exploded):
            color = (_theme.LIGHT if k == selected
                     else _theme.ACCENTS[k % len(_theme.ACCENTS)])
            plotter.add_mesh(block, name=f"chunk {k}", color=color)
        label = ("click a chunk: closest() names it" if selected is None
                 else f"chunk {selected}")
        plotter.add_text(label, name="picked", position="lower_left",
                         font_size=12, color=_theme.LIGHT)

    def name_chunk(point):
        found = closest(exploded, point)
        print(f"chunk {found.block_index} "
              f"(witness distance {found.distance:.2e})")
        draw_chunks(found.block_index)

    draw_chunks(None)
    plotter.enable_surface_point_picking(
        callback=name_chunk, left_clicking=True, show_message=False,
        color=_theme.LIGHT, point_size=12)
    plotter.view_vector((0.55, -1.0, 0.4), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
