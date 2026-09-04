"""
Signed distance recolored live under a dragged torus probe

A hills surface wears its signed euclidean distance to a closed torus
probe as per-vertex color on the Polydera diverging ramp: negative
inside the tube toward the warm pole, positive outside toward teal, the
zero seam drawn where the probe pierces the terrain. Where the sunken
ring passes through the hills the warm region is an annulus around a
teal island in the hole, and dragging slides that ring across the
terrain. The small sphere widget at the ring's center is the drag
handle — every drag hands its center to the callback, which rebuilds
the torus there and asks the accessor for the field again over all 122k
surface points. The color limits stay fixed at the tube radius — the
deepest any point can sit inside the tube — so the warm pole saturates
exactly at the tube core and dragging never rescales the ramp.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401

CENTER = (0.0, 10.0, 3.0)
RING = 3.0     # centerline radius
TUBE = 1.2     # cross-section radius, the deepest inside distance
HANDLE = 0.9


def _probe(center):
    return pv.ParametricTorus(ringradius=RING, crosssectionradius=TUBE,
                              u_res=48, v_res=24).translate(center)


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
                     cmap=_theme.polydera_div(), clim=(-TUBE, TUBE))

    def draw(probe):
        plotter.add_mesh(probe, name="probe", color=_theme.ROSE,
                         style="wireframe", line_width=1.0, opacity=0.5,
                         render_lines_as_tubes=False, lighting=False)

    def resample(center):
        probe = _probe(center)
        draw(probe)
        hills.point_data["signed distance"] = \
            hills.trueform.signed_distance(probe)

    draw(_probe(CENTER))
    plotter.add_sphere_widget(resample, center=CENTER, radius=HANDLE,
                              theta_resolution=16, phi_resolution=16,
                              color=_theme.ROSE, style="wireframe",
                              interaction_event="always")
    plotter.add_text("drag the handle: signed distance recolors live",
                     position="lower_left", font_size=12,
                     color=_theme.LIGHT)
    plotter.view_vector((0.5, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
    plotter.camera.zoom(1.25)
    plotter.show()


if __name__ == "__main__":
    main()
