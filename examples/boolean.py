"""
Boolean difference recut live from a dragged sphere

A cube loses a spherical bite, and the result wears its provenance: the
accessor's difference carries source labels as cell data, so the bite
wall renders in the cutter's color. The sphere is a standard PyVista
sphere widget — every drag hands its center to the callback, which
rebuilds the cutter there and asks for the difference again. The cube's
cached trueform mesh amortizes across drags, so the boolean recuts
fluidly under the mouse.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401

CENTER = (0.5, 0.3, 0.3)
RADIUS = 0.6


def _cutter(center):
    return pv.Sphere(radius=RADIUS, center=center,
                     theta_resolution=48, phi_resolution=48)


def compute(center=CENTER):
    base = pv.Cube()
    carved = base.trueform.difference(_cutter(center))
    return base, carved


def main():
    import _theme
    base, carved = compute()
    print(f"base {base.n_cells} faces -> carved {carved.n_cells} faces")
    plotter = pv.Plotter(theme=_theme.theme())

    def draw(result):
        plotter.add_mesh(result, name="carved", scalars="trueform_labels",
                         cmap=_theme.label_cmap(2), clim=(0, 1),
                         show_scalar_bar=False)

    def recut(center):
        draw(base.trueform.difference(_cutter(center)))

    draw(carved)
    plotter.add_sphere_widget(recut, center=CENTER, radius=RADIUS,
                              theta_resolution=48, phi_resolution=48,
                              color=_theme.ROSE, style="wireframe",
                              interaction_event="always")
    plotter.add_text("drag the sphere: the difference recuts live",
                     position="lower_left", font_size=12,
                     color=_theme.LIGHT)
    plotter.view_vector((0.6, -1.0, 0.5), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.3)
    plotter.show()


if __name__ == "__main__":
    main()
