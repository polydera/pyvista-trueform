"""
The remesh tier side by side

One sphere through trueform's three surface tiers: isotropic remeshing
toward a target edge length, quadric-error decimation to a face budget,
and quadric-error simplification to an error budget. Face counts and mean
edge lengths are printed, and the four surfaces render side by side with
their wireframes.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv
import trueform as tf

import pyvista_trueform  # registers the accessor  # noqa: F401

NAMES = ("input", "remeshed", "decimated", "simplified")


def compute():
    sphere = pv.Sphere()
    target = 1.5 * tf.mean_edge_length(sphere.trueform.to_mesh())
    remeshed = sphere.trueform.remeshed(target)
    decimated = sphere.trueform.decimated(0.3)
    simplified = sphere.trueform.simplified()
    return sphere, remeshed, decimated, simplified


def main():
    variants = compute()
    for name, mesh in zip(NAMES, variants):
        length = tf.mean_edge_length(mesh.trueform.to_mesh())
        print(f"{name}: {mesh.n_cells} faces, mean edge length {length:.4f}")
    plotter = pv.Plotter(shape=(1, 4))
    for k, (name, mesh) in enumerate(zip(NAMES, variants)):
        plotter.subplot(0, k)
        plotter.add_mesh(mesh, show_edges=True)
        plotter.add_text(name)
    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    main()
