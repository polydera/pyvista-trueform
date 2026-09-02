"""
An orthographic depth image from one batched ray cast

One trueform Ray carries a whole orthographic camera — a grid of origins
above a random-hills surface, all looking straight down — and a single
mesh.trueform.ray_cast answers every pixel at once through the cached
spatial tree: the returned ray parameters ARE the depth image, NaN where
a ray misses. The surface and its depth map render side by side, the ray
direction drawn over the surface.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv
import trueform as tf

import pyvista_trueform  # registers the accessor  # noqa: F401


def compute(resolution=256):
    surface = pv.ParametricRandomHills()
    dtype = surface.trueform.to_mesh().dtype
    points = np.asarray(surface.points)
    lo, hi = points.min(axis=0), points.max(axis=0)

    x = np.linspace(lo[0], hi[0], resolution, dtype=dtype)
    y = np.linspace(hi[1], lo[1], resolution, dtype=dtype)
    gx, gy = np.meshgrid(x, y)
    origins = np.zeros((resolution * resolution, 3), dtype=dtype)
    origins[:, 0] = gx.ravel()
    origins[:, 1] = gy.ravel()
    origins[:, 2] = hi[2] + 1.0
    directions = np.zeros((resolution * resolution, 3), dtype=dtype)
    directions[:, 2] = -1.0

    rays = tf.Ray(origin=origins, direction=directions)
    _, ts = surface.trueform.ray_cast(rays)
    return ts.reshape(resolution, resolution)


def main():
    import _theme
    depth = compute()
    hits = ~np.isnan(depth)
    print(f"{int(hits.sum()):,} hits of {depth.size:,} rays, depth "
          f"{np.nanmin(depth):.3f} .. {np.nanmax(depth):.3f}")

    surface = pv.ParametricRandomHills()
    points = np.asarray(surface.points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    resolution = depth.shape[0]
    image = pv.ImageData(
        dimensions=(resolution, resolution, 1),
        spacing=((hi[0] - lo[0]) / (resolution - 1),
                 (hi[1] - lo[1]) / (resolution - 1), 1.0),
        origin=(lo[0], lo[1], 0.0))
    image.point_data["ray parameter t"] = np.flipud(depth).ravel()

    plotter = pv.Plotter(shape=(1, 2), theme=_theme.theme())
    plotter.subplot(0, 0)
    plotter.add_mesh(surface, color=_theme.TEAL)
    plotter.add_mesh(pv.Arrow(start=(0.0, 0.0, hi[2] + 3.5),
                              direction=(0.0, 0.0, -1.0), scale=2.5),
                     color=_theme.ROSE)
    plotter.add_text("one batched tf.Ray, straight down",
                     position="lower_left", font_size=12,
                     color=_theme.LIGHT)
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.45)
    plotter.subplot(0, 1)
    plotter.add_mesh(image, scalars="ray parameter t",
                     cmap=_theme.polydera_cmap(depth),
                     nan_color=_theme.BACKGROUND,
                     lighting=False,
                     scalar_bar_args=dict(title="ray parameter t"))
    plotter.view_xy()
    plotter.camera.zoom(1.3)
    plotter.show()


if __name__ == "__main__":
    main()
