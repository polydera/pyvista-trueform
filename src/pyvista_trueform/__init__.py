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
from ._accessor import TrueformAccessor
from ._conversion import (curves_to_pyvista, domains_to_pyvista,
                          to_pyvista, to_trueform)
from ._factory import CsgGraph, csg_graph, domains, mesh_arrangements
from ._io import read, write
from ._registration import align, chamfer_distance
from ._tube import tube

__all__ = [
    "CsgGraph",
    "TrueformAccessor",
    "align",
    "chamfer_distance",
    "csg_graph",
    "curves_to_pyvista",
    "domains",
    "domains_to_pyvista",
    "mesh_arrangements",
    "read",
    "to_pyvista",
    "to_trueform",
    "tube",
    "write",
]
