"""
The Polydera colormaps, registered with matplotlib on import

The sequential and diverging maps are the Polydera maps defined in
lunar's src/viewport/colorMaps.ts; the waypoints here mirror that file,
and this module is their one producer on the Python side. Importing the
package registers both, so ``cmap="polydera"`` and ``cmap="polydera_div"``
resolve anywhere matplotlib accepts a colormap name.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

_BACKGROUND = "#0c1513"
_SEQ_WAYPOINTS = ("#001a17", "#006d63", "#00d5be", "#ffbb7a", "#ff6b2c")
_DIV_WAYPOINTS = ("#ff6b2c", "#a0401a", "#606060", "#1a5a52", "#00d5be")


def _waypoint_cmap(name, waypoints):
    cmap = LinearSegmentedColormap.from_list(name, waypoints)
    cmap.set_bad(_BACKGROUND)
    return cmap


def polydera_seq():
    """The Polydera sequential map: dark teal ground to brand teal to warm."""
    return _waypoint_cmap("polydera", _SEQ_WAYPOINTS)


def polydera_div():
    """The Polydera diverging map: orange through mid gray (zero) to teal.

    The center gray is deliberately not the background color, so zero
    stays legible instead of vanishing into the page.
    """
    return _waypoint_cmap("polydera_div", _DIV_WAYPOINTS)


def polydera_cmap(values):
    """Diverging when ``values`` cross zero, sequential otherwise."""
    values = np.asarray(values)
    if np.nanmin(values) < 0.0 < np.nanmax(values):
        return polydera_div()
    return polydera_seq()


def _register():
    for cmap in (polydera_seq(), polydera_div()):
        if cmap.name not in matplotlib.colormaps:
            matplotlib.colormaps.register(cmap)


_register()

__all__ = ["polydera_cmap", "polydera_div", "polydera_seq"]
