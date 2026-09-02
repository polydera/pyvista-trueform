"""
Registration recovering a known motion

A surface is moved by a known rotation and translation, then registered
back onto the original with align(method="rigid") — the moved copy keeps
point-to-point correspondence, which is exactly what the rigid (Kabsch)
fit wants. The chamfer distance before and after shows the recovery, and
the three surfaces render together.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import pyvista as pv

from pyvista_trueform import align, chamfer_distance


def _applied(matrix, points):
    return (points @ matrix[:3, :3].T.astype(points.dtype)
            + matrix[:3, 3].astype(points.dtype))


def compute():
    target = pv.ParametricRandomHills()
    angle = np.radians(25.0)
    motion = np.eye(4, dtype=np.float32)
    motion[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]
    motion[:3, 3] = [2.0, -1.0, 0.5]
    moved = target.copy()
    moved.points = _applied(motion, np.asarray(target.points))

    before = chamfer_distance(moved, target)
    recovered = align(moved, target, method="rigid")
    aligned = moved.copy()
    aligned.points = _applied(recovered, np.asarray(moved.points))
    after = chamfer_distance(aligned, target)
    return target, moved, aligned, before, after


def main():
    import _theme
    target, moved, aligned, before, after = compute()
    print(f"chamfer before: {before:.4f}, after: {after:.2e}")
    plotter = pv.Plotter(theme=_theme.theme())
    plotter.add_mesh(target, color=_theme.LIGHT, style="wireframe",
                     opacity=0.3, line_width=1,
                     render_lines_as_tubes=False)
    plotter.add_mesh(moved, color=_theme.ROSE, opacity=0.45)
    plotter.add_mesh(aligned, color=_theme.TEAL)
    plotter.add_text(f"chamfer {before:.3f} -> {after:.2e}",
                     position="lower_left", font_size=12,
                     color=_theme.LIGHT)
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
