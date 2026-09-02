"""
Isobands and isocontours of a height field

A random-hills surface carries its own height as a per-vertex scalar
field. The accessor recuts the mesh into height bands — every face split
exactly at the cut values, the band of each output face riding as cell
data — and extracts the isocontour curves at the same values, overlaid on
the banded surface. A slider recuts live: the surface never changes, so
the accessor's cached mesh is reused and every drag pays only the recut.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401


def compute(n_bands=7):
    hills = pv.ParametricRandomHills()
    height = np.asarray(hills.points)[:, 2].copy()
    hills.point_data["height"] = height
    cuts = np.linspace(height.min(), height.max(), n_bands + 1)[1:-1]
    bands = hills.trueform.isobands("height", cuts)
    contours = hills.trueform.isocontours("height", cuts)
    return bands, contours


def main():
    import _theme
    hills = pv.ParametricRandomHills()
    height = np.asarray(hills.points)[:, 2].copy()
    hills.point_data["height"] = height
    lo, hi = float(height.min()), float(height.max())

    plotter = pv.Plotter(theme=_theme.theme())

    def recut(n_bands):
        n = max(2, int(round(n_bands)))
        cuts = np.linspace(lo, hi, n + 1)[1:-1]
        bands = hills.trueform.isobands("height", cuts)
        contours = hills.trueform.isocontours("height", cuts)
        plotter.add_mesh(bands, name="bands", scalars="trueform_labels",
                         cmap=_theme.polydera_bands(n),
                         show_scalar_bar=False)
        plotter.add_mesh(contours, name="contours", color=_theme.LIGHT,
                         line_width=5, render_lines_as_tubes=True)

    recut(7)
    plotter.add_slider_widget(recut, rng=[2, 15], value=7, title="bands",
                              fmt="%.0f", color=_theme.LIGHT,
                              interaction_event="always")
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
