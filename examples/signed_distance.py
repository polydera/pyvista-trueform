"""
Signed distance recolored live under a dragged probe

A hills surface wears its signed euclidean distance to a probe sphere
as per-vertex color on the Polydera diverging ramp: negative inside the
probe toward the warm pole, positive outside toward teal, the zero seam
drawn where the probe pierces the terrain. The probe is a standard
PyVista sphere widget — every drag hands its center to the callback,
which rebuilds the probe there and asks the accessor for the field
again over all 122k surface points. The color limits stay fixed and
symmetric about zero, so dragging moves the seam without ever rescaling
the ramp.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401

CENTER = (0.0, 10.0, 3.0)
RADIUS = 3.0


def _probe(center):
    return pv.Sphere(radius=RADIUS, center=center,
                     theta_resolution=32, phi_resolution=32)


def compute(center=CENTER):
    hills = pv.ParametricRandomHills(u_res=350, v_res=350)
    field = hills.trueform.signed_distance(_probe(center))
    return hills, field


def main():
    import _theme
    hills, field = compute()
    print(f"{hills.n_points} points, signed distance "
          f"{field.min():.2f} .. {field.max():.2f}")
    hills.point_data["signed distance"] = field
    plotter = pv.Plotter(theme=_theme.theme())
    plotter.add_mesh(hills, scalars="signed distance",
                     cmap=_theme.polydera_div(), clim=(-RADIUS, RADIUS))

    def resample(center):
        hills.point_data["signed distance"] = \
            hills.trueform.signed_distance(_probe(center))

    plotter.add_sphere_widget(resample, center=CENTER, radius=RADIUS,
                              theta_resolution=32, phi_resolution=32,
                              color=_theme.ROSE, style="wireframe",
                              interaction_event="always")
    plotter.add_text("drag the probe: signed distance recolors live",
                     position="lower_left", font_size=12,
                     color=_theme.LIGHT)
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
