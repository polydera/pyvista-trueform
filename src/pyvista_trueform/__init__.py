"""
PyVista integration for trueform: conversions, the .trueform accessor, and
N-ary CSG

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

# Importing _accessor registers the .trueform accessor on pyvista.PolyData.
# Installed wheels also reach it through PyVista's lazy accessor entry point.
try:
    from importlib.metadata import version as _distribution_version
    __version__ = _distribution_version("pyvista-trueform")
except Exception:
    __version__ = "0.0.0"

from ._accessor import TrueformAccessor
from ._colormaps import polydera_cmap, polydera_div, polydera_seq
from ._conversion import (curves_to_pyvista, domains_to_pyvista,
                          to_pyvista, to_trueform)
from ._factory import (CsgGraph, csg_graph, domains, mesh_arrangements,
                       split_into_domains)
from ._generators import box, cylinder, plane, sphere
from ._io import read, write
from ._lines import connect_lines
from ._pick import ClosestHit, RayHit, closest, pick
from ._registration import (align_icp, align_knn, align_obb, align_rigid,
                            align_similarity, chamfer_distance)
from ._tube import tube

__all__ = [
    "__version__",
    "ClosestHit",
    "CsgGraph",
    "RayHit",
    "TrueformAccessor",
    "align_icp",
    "align_knn",
    "align_obb",
    "align_rigid",
    "align_similarity",
    "box",
    "chamfer_distance",
    "closest",
    "connect_lines",
    "csg_graph",
    "curves_to_pyvista",
    "cylinder",
    "domains",
    "domains_to_pyvista",
    "mesh_arrangements",
    "pick",
    "plane",
    "polydera_cmap",
    "polydera_div",
    "polydera_seq",
    "read",
    "sphere",
    "split_into_domains",
    "to_pyvista",
    "to_trueform",
    "tube",
    "write",
]
