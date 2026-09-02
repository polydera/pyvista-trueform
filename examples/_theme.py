"""
The Polydera color scheme and presentation theme for the example gallery

One dark ground, one teal, and six categorical accents, shared by every
example's plotting path. The theme is handed to each Plotter explicitly —
never installed globally — so importing an example colors nothing. The
sequential and diverging colormaps are the Polydera maps defined in
lunar's src/viewport/colorMaps.ts; the waypoints here mirror that file.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

BACKGROUND = "#0c1513"
TEAL = "#00d5be"
ROSE = "#c44569"
AMBER = "#e0a23b"
CORAL = "#d97559"
PURPLE = "#9573c4"
BLUE = "#5985c4"
LIGHT = "#e8f2ef"
EDGE = "#1e4038"
ACCENTS = (TEAL, ROSE, AMBER, CORAL, PURPLE, BLUE)

SEQ_WAYPOINTS = ("#001a17", "#006d63", "#00d5be", "#ffbb7a", "#ff6b2c")
DIV_WAYPOINTS = ("#ff6b2c", "#a0401a", "#606060", "#1a5a52", "#00d5be")


def theme():
    """A configured :class:`pyvista.themes.Theme` for one Plotter."""
    t = pv.themes.Theme()
    t.background = BACKGROUND
    t.color = TEAL
    t.edge_color = EDGE
    t.outline_color = LIGHT
    t.font.color = LIGHT
    t.smooth_shading = True
    t.split_sharp_edges = True
    t.lighting_params.interpolation = "Phong"
    t.lighting_params.ambient = 0.32
    t.lighting_params.diffuse = 0.68
    t.lighting_params.specular = 0.18
    t.lighting_params.specular_power = 28.0
    t.render_lines_as_tubes = True
    t.line_width = 3
    t.anti_aliasing = "msaa"
    t.multi_samples = 8
    return t


def label_cmap(n):
    """A ListedColormap cycling the six accents over ``n`` labels."""
    from matplotlib.colors import ListedColormap

    return ListedColormap([ACCENTS[k % len(ACCENTS)] for k in range(n)])


def _waypoint_cmap(name, waypoints):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(name, waypoints)
    cmap.set_bad(BACKGROUND)
    return cmap


def polydera_seq():
    """The Polydera sequential map: dark teal ground to brand teal to warm."""
    return _waypoint_cmap("polydera_seq", SEQ_WAYPOINTS)


def polydera_div():
    """The Polydera diverging map: orange through mid gray (zero) to teal."""
    return _waypoint_cmap("polydera_div", DIV_WAYPOINTS)


def polydera_cmap(values):
    """Diverging when ``values`` cross zero, sequential otherwise."""
    import numpy as np

    values = np.asarray(values)
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        return polydera_div()
    return polydera_seq()
