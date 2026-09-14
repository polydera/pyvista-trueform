"""
A mesh's own distance field, read back as nested offset surfaces

The accessor samples the torus's signed distance onto a grid in banded
mode: only the samples within `band` voxels of the surface are measured
and the far field is swept from them, an order of magnitude faster with
the sign exact everywhere. That band is the example's one real
constraint — an offset surface is the level set at `+d`, so the band has
to reach at least `d / spacing` voxels or the shell being drawn stands on
swept values rather than measured ones. Here it is sized from the
largest offset asked for.

Each shell is the same field read at another isovalue, never a rebuilt
field, and the progression is the point: once the offset closes the gap
between the tube and the ring, the hole is gone and the surface is one
solid lobe — a torus by Euler characteristic for the first two shells, a
sphere for the third. The input rides along as a wireframe, in the same
coordinates: the grid samples the dataset's own space, so nothing is
registered afterwards.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv

import pyvista_trueform  # registers the accessors  # noqa: F401

RING, TUBE = 1.0, 0.35
OFFSETS = (0.15, 0.4, 0.7)  # the last one past RING - TUBE: the hole seals


def _grid(bounds, margin, resolution):
    """A cubic grid around `bounds`, clear of the surface by `margin`."""
    low = np.asarray(bounds[::2]) - 1.3 * margin
    high = np.asarray(bounds[1::2]) + 1.3 * margin
    step = float((high - low).max()) / (resolution - 1)
    dims = tuple(int(np.ceil(extent / step)) + 1 for extent in high - low)
    return dims, (step,) * 3, tuple(low)


def compute(resolution=80, offsets=OFFSETS):
    torus = pv.ParametricTorus(ringradius=RING, crosssectionradius=TUBE,
                               u_res=60, v_res=30)
    dims, spacing, origin = _grid(torus.bounds, max(offsets), resolution)
    band = int(np.ceil(max(offsets) / spacing[0])) + 2
    field = torus.trueform.signed_distance_field(
        dims, spacing, origin, mode="banded", band=band)
    return torus, field, [field.trueform.isosurface(d) for d in offsets]


def main():
    import _theme
    torus, field, shells = compute()
    print(f"{field.dimensions} samples, offsets "
          + ", ".join(f"{d:.2f} -> {s.n_cells} faces"
                      for d, s in zip(OFFSETS, shells)))

    plotter = pv.Plotter(theme=_theme.theme())
    plotter.add_mesh(torus, color=_theme.LIGHT, style="wireframe",
                     line_width=1.0, render_lines_as_tubes=False,
                     lighting=False)
    for shell, color, opacity in zip(shells, _theme.ACCENTS,
                                     (0.85, 0.45, 0.25)):
        plotter.add_mesh(shell, color=color, opacity=opacity)
    plotter.add_text("one banded field, read at three offsets",
                     position="lower_left", font_size=12, color=_theme.LIGHT)
    plotter.view_vector((1.0, -1.0, 0.6), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.3)
    plotter.show()


if __name__ == "__main__":
    main()
