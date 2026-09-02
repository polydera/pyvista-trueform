"""
An orthographic depth image from one batched ray cast

One trueform Ray carries a whole orthographic camera — a grid of origins
above a random-hills surface, all looking straight down — and a single
mesh.trueform.ray_cast answers every pixel at once through the cached
spatial tree: the returned ray parameters ARE the depth image, NaN where
a ray misses. Shown with matplotlib when available, summarized otherwise.

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
    depth = compute()
    hits = ~np.isnan(depth)
    print(f"{int(hits.sum()):,} hits of {depth.size:,} rays, depth "
          f"{np.nanmin(depth):.3f} .. {np.nanmax(depth):.3f}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("install matplotlib for the depth image")
        return
    plt.imshow(depth, cmap="viridis")
    plt.colorbar(label="ray parameter t")
    plt.axis("off")
    plt.title("one batched tf.Ray, one ray_cast call")
    plt.show()


if __name__ == "__main__":
    main()
