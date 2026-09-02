"""
PyVista integration for trueform: conversions, the .trueform accessor, and
N-ary CSG

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

from ._conversion import curves_to_pyvista, to_pyvista, to_trueform

__all__ = [
    "curves_to_pyvista",
    "to_pyvista",
    "to_trueform",
]
