"""
A solid carved as a scalar field, surfaced with its creases intact

A box's signed distance field minus a sphere's is a new field whose zero
level set is exactly the carved solid — sampled geometry, not cut
geometry, so the operands never meet as triangles. Surfacing it twice
shows what the field costs and what the extractor recovers: flying edges
places a vertex per crossed grid edge and rounds every crease off to the
grid, while dual contouring fits one vertex per cell to the field's own
crossings and keeps the box's edges and the carve's rim at the same
resolution. A slider moves the isovalue rather than rebuilding anything:
each stop is the offset surface of the one field, recut off the volume
accessor's cached field.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

import pyvista_trueform as tfpv

EXTENT = 0.9            # the grid reaches this far from the origin
BALL = (0.5, 0.5, 0.5)  # the carving sphere, biting one corner of the box
RADIUS = 0.55


def _grid(resolution):
    spacing = 2.0 * EXTENT / (resolution - 1)
    return ((resolution,) * 3, (spacing,) * 3, (-EXTENT,) * 3)


def compute(resolution=64, iso=0.0):
    box = pv.Cube().triangulate()
    dims, spacing, origin = _grid(resolution)
    field = box.trueform.signed_distance_field(dims, spacing, origin)
    carved = field.trueform.difference(
        tfpv.sphere_sdf(dims, spacing, origin, BALL, RADIUS))
    rounded = carved.trueform.isosurface(iso)
    sharp = carved.trueform.isosurface(iso, method="dual_contouring")
    return carved, rounded, sharp


def main():
    import _theme
    carved, rounded, sharp = compute()
    print(f"{carved.n_points} samples, flying edges {rounded.n_cells} faces, "
          f"dual contouring {sharp.n_cells} faces")

    plotter = pv.Plotter(theme=_theme.theme(), shape=(1, 2))

    def resurface(iso):
        plotter.subplot(0, 0)
        plotter.add_mesh(carved.trueform.isosurface(iso), name="rounded",
                         color=_theme.TEAL, show_edges=True)
        plotter.subplot(0, 1)
        plotter.add_mesh(
            carved.trueform.isosurface(iso, method="dual_contouring"),
            name="sharp", color=_theme.AMBER, show_edges=True)

    plotter.subplot(0, 0)
    plotter.add_text("flying edges", position="upper_edge", font_size=11,
                     color=_theme.LIGHT)
    plotter.subplot(0, 1)
    plotter.add_text("dual contouring", position="upper_edge", font_size=11,
                     color=_theme.LIGHT)
    resurface(0.0)
    plotter.add_slider_widget(resurface, rng=[-0.15, 0.25], value=0.0,
                              title="isovalue", color=_theme.LIGHT,
                              interaction_event="always")
    plotter.link_views()
    plotter.view_vector((1.0, -1.0, 0.7), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.2)
    plotter.show()


if __name__ == "__main__":
    main()
