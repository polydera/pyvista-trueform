"""
Surfaces colored by curvature

The accessor answers per-vertex principal curvatures in one call, and
the two classical reads render side by side on the theme's diverging
ramp, centered at zero: Gaussian curvature (the product) changes sign
across a torus — positive on the outer half, negative on the inner —
and mean curvature (the average) splits a random-hills surface into its
crests and its bowls.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401


def compute():
    torus = pv.ParametricTorus()
    k0, k1 = torus.trueform.principal_curvatures()
    gaussian = k0 * k1
    hills = pv.ParametricRandomHills()
    h0, h1 = hills.trueform.principal_curvatures()
    mean = 0.5 * (h0 + h1)
    return torus, gaussian, hills, mean


def main():
    import _theme
    torus, gaussian, hills, mean = compute()
    print(f"gaussian curvature {gaussian.min():.3f} .. {gaussian.max():.3f}, "
          f"mean curvature {mean.min():.3f} .. {mean.max():.3f}")
    torus.point_data["gaussian curvature"] = gaussian
    hills.point_data["mean curvature"] = mean
    plotter = pv.Plotter(shape=(1, 2), theme=_theme.theme())
    plotter.subplot(0, 0)
    limit = float(np.percentile(np.abs(gaussian), 98.0))
    plotter.add_mesh(torus, scalars="gaussian curvature",
                     cmap=_theme.polydera_cmap(gaussian),
                     clim=(-limit, limit))
    plotter.view_vector((0.35, -0.85, 0.9), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.35)
    plotter.subplot(0, 1)
    limit = float(np.percentile(np.abs(mean), 98.0))
    plotter.add_mesh(hills, scalars="mean curvature",
                     cmap=_theme.polydera_cmap(mean), clim=(-limit, limit))
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
