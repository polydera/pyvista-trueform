"""
The remesh tier side by side

One sphere through trueform's three surface tiers: isotropic remeshing
toward a target edge length, quadric-error decimation to a face budget,
and quadric-error simplification to an error budget calibrated here to
land near the decimation's face count, so the grid compares four views
of similar coarseness. Face counts and mean edge lengths are printed,
and the four surfaces render in a 2x2 grid with their wireframes, the
cameras linked.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv
import trueform as tf

import pyvista_trueform  # registers the accessor  # noqa: F401

NAMES = ("input", "remeshed", "decimated", "simplified")
PARAMETERS = ("", "target 1.5x mean edge", "0.3 of the faces",
              "error 1% of the diagonal")


def compute():
    sphere = pv.Sphere()
    target = 1.5 * tf.mean_edge_length(sphere.trueform.to_mesh())
    remeshed = sphere.trueform.remeshed(target)
    decimated = sphere.trueform.decimated(0.3)
    simplified = sphere.trueform.simplified(error_rel=0.01)
    return sphere, remeshed, decimated, simplified


def main():
    import _theme
    variants = compute()
    for name, mesh in zip(NAMES, variants):
        length = tf.mean_edge_length(mesh.trueform.to_mesh())
        print(f"{name}: {mesh.n_cells} faces, mean edge length {length:.4f}")
    plotter = pv.Plotter(shape=(2, 2), theme=_theme.theme())
    for k, (name, parameter, mesh) in enumerate(
            zip(NAMES, PARAMETERS, variants)):
        plotter.subplot(k // 2, k % 2)
        plotter.add_mesh(mesh, show_edges=True,
                         color=_theme.TEAL, edge_color=_theme.EDGE)
        label = f"{name} — {mesh.n_cells} faces"
        if parameter:
            label += f"\n{parameter}"
        plotter.add_text(label, font_size=12, color=_theme.LIGHT)
    plotter.link_views()
    plotter.view_vector((0.4, -1.0, 0.4), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.5)
    plotter.show()


if __name__ == "__main__":
    main()
