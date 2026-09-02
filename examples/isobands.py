"""
Isobands and isocontours of a height field

A random-hills surface carries its own height as a per-vertex scalar
field. The accessor recuts the mesh into height bands — every face split
exactly at the cut values, the band of each output face riding as cell
data — and extracts the isocontour curves at the same values, overlaid on
the banded surface.

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
    bands, contours = compute()
    print(f"{bands.n_cells} band faces, "
          f"{contours.GetNumberOfLines()} contour lines")
    plotter = pv.Plotter()
    plotter.add_mesh(bands, scalars="trueform_labels", cmap="terrain",
                     show_scalar_bar=False)
    plotter.add_mesh(contours, color="black", line_width=4,
                     render_lines_as_tubes=True)
    plotter.show()


if __name__ == "__main__":
    main()
