"""
A field's level sets on a plane, drawn beside the surface they cut

The field is two analytic sphere distance fields unioned into one solid,
and the plane cuts it lengthwise. `slice_contours` resamples the volume
once onto the plane's own 2D grid and contours every isovalue on that
slice, welding the crossings into connected polylines — so the answer is
curves with topology, not a marching soup, and it comes back as line
PolyData in the same space as the isosurface it is drawn against.

The isovalues are read off one field, and the topology is what they
show: the zero level set traces where the surface meets the plane as one
curve, the negative one has already parted into a ring around each lobe,
and the positive one floats outside both as the offset would. The plane
is stated in the grid's own space, and the contours terminate cleanly at
the volume's boundary rather than wrapping around it.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv

import pyvista_trueform as tfpv

EXTENT = 2.2                    # the grid reaches this far from the origin
LOBES = ((-0.8, 0.0, 0.0), (0.8, 0.0, 0.0))
RADIUS = 1.0
ISOVALUES = (-0.35, 0.0, 0.4)


def compute(resolution=64, isovalues=ISOVALUES):
    spacing = 2.0 * EXTENT / (resolution - 1)
    grid = ((resolution,) * 3, (spacing,) * 3, (-EXTENT,) * 3)
    field = tfpv.sphere_sdf(*grid, LOBES[0], RADIUS).trueform.union(
        tfpv.sphere_sdf(*grid, LOBES[1], RADIUS))

    nodes = 2 * resolution
    contours = field.trueform.slice_contours(
        (-EXTENT, -EXTENT, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (nodes, nodes), (2.0 * EXTENT / (nodes - 1),) * 2, isovalues)
    return field, field.trueform.isosurface(), contours


def main():
    import _theme
    field, surface, contours = compute()
    print(f"{field.dimensions} samples -> {surface.n_cells} faces, "
          f"{contours.GetNumberOfLines()} contours over "
          f"{contours.n_points} welded points")

    plotter = pv.Plotter(theme=_theme.theme())
    plotter.add_mesh(surface, color=_theme.TEAL, opacity=0.25)
    plotter.add_mesh(pv.Plane(center=(0.0, 0.0, 0.0), direction=(0, 0, 1),
                              i_size=2 * EXTENT, j_size=2 * EXTENT),
                     color=_theme.LIGHT, opacity=0.08, lighting=False)
    plotter.add_mesh(tfpv.tube(contours, radius=0.02), color=_theme.AMBER)
    plotter.add_text(
        "level sets " + ", ".join(f"{v:+.2f}" for v in ISOVALUES)
        + " on one plane", position="lower_left", font_size=12,
        color=_theme.LIGHT)
    plotter.view_vector((0.7, -1.0, 0.8), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.3)
    plotter.show()


if __name__ == "__main__":
    main()
