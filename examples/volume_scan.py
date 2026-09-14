"""
A scan read in patient space, surfaced at a Hounsfield threshold

NIfTI-1 in, PolyData out. The file's affine lands on the grid as its
origin, spacing and direction, so the level set the accessor extracts is
already a patient-space mesh — nothing is transformed afterwards and no
placement is rederived here. Run it against a real scan by passing the
path (`python examples/volume_scan.py ct.nii.gz`); with no argument it
synthesizes a small int16 study — a shell of dense tissue on a scanner's
own 0.7 x 0.7 x 1.5 mm grid, tilted and shifted the way a patient lies —
writes it through the suffix-dispatched `write`, and reads it back, so
the affine round trip is part of what runs.

The samples stay int16 across all of it: the measurement is not widened
for having crossed a file, and the emitted points are float32 because
that is the grid the counts stand on. A corner is inside when
`sample < iso` and dense tissue is the HIGH side, so this surface winds
into the air around it; negate the field into a signed one where outward
normals matter.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv

import pyvista_trueform as tfpv

THRESHOLD = 300.0          # Hounsfield units: dense tissue against soft
SPACING = (0.7, 0.7, 1.5)  # the scanner's own millimetres, thicker along z
AIR, SOFT, DENSE = -1000, 40, 1200


def _study(dims=(64, 64, 36)):
    """A small int16 study: a shell of dense tissue, posed like a patient."""
    index = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in dims),
                        indexing="ij")
    radius = np.sqrt(sum(((i - (n - 1) / 2) / (0.42 * n)) ** 2
                         for i, n in zip(index, dims)))
    counts = np.full(dims, AIR, dtype=np.int16, order="F")
    counts[radius <= 1.0] = SOFT
    counts[(radius > 0.82) & (radius <= 1.0)] = DENSE

    scan = pv.ImageData(dimensions=dims, spacing=SPACING)
    scan.point_data["counts"] = counts.ravel(order="F")
    tilt = np.radians(18.0)
    scan.direction_matrix = [[1.0, 0.0, 0.0],
                             [0.0, np.cos(tilt), -np.sin(tilt)],
                             [0.0, np.sin(tilt), np.cos(tilt)]]
    scan.origin = (-22.0, -18.0, 140.0)  # where the table puts the study
    return scan


def compute(path=None):
    if path is not None:
        scan = tfpv.read(path)
    else:
        with tempfile.TemporaryDirectory() as directory:
            written = Path(directory) / "study.nii.gz"
            tfpv.write(written, _study())
            scan = tfpv.read(written)
    return scan, scan.trueform.isosurface(THRESHOLD)


def main():
    import _theme
    scan, surface = compute(sys.argv[1] if len(sys.argv) > 1 else None)
    posed = not np.allclose(scan.direction_matrix, np.eye(3))
    print(f"{scan.dimensions} samples of {scan.active_scalars.dtype}, "
          f"spacing {tuple(round(s, 3) for s in scan.spacing)}, "
          f"{'posed' if posed else 'axis-aligned'} -> {surface.n_cells} faces")

    plotter = pv.Plotter(theme=_theme.theme())
    plotter.add_mesh(surface, color=_theme.LIGHT, opacity=0.45)
    plotter.add_mesh(scan.outline(), color=_theme.TEAL, line_width=2)
    plotter.add_axes(color=_theme.LIGHT)
    plotter.add_text(f"{THRESHOLD:.0f} HU in patient space",
                     position="lower_left", font_size=12, color=_theme.LIGHT)
    plotter.view_vector((1.0, -1.0, 0.45), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.2)
    plotter.show()


if __name__ == "__main__":
    main()
