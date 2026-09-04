"""
The Polydera color scheme and presentation theme for the example gallery

One dark ground, one teal, and six categorical accents, shared by every
example's plotting path. The theme is handed to each Plotter explicitly —
never installed globally — so importing an example colors nothing. The
sequential and diverging colormaps come from pyvista_trueform itself —
the package is the one producer of the Polydera maps.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import pyvista as pv

from pyvista_trueform import (polydera_cmap, polydera_div,  # noqa: F401
                              polydera_seq)

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


def polydera_bands(n, floor=0.15):
    """``n`` sequential band colors, starting visibly above the page ground.

    The sequential map's first waypoint nearly matches the background, so a
    band colored at t=0 reads as a hole in the surface; the floor keeps the
    lowest band on the page.
    """
    import numpy as np
    from matplotlib.colors import ListedColormap

    return ListedColormap(polydera_seq()(np.linspace(floor, 1.0, n)))
