"""
The package contract, queryable at runtime

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

from importlib.resources import files


def agents():
    """The package's caller-facing contract, as one markdown string.

    The AGENTS.md convention transplanted to runtime: the dialect laws,
    the full public surface with signatures and return shapes in PyVista
    terms, and the contracts that bite, so an agent driving the package
    in a live session can query them without leaving it —
    ``print(pyvista_trueform.agents())``.
    """
    return files("pyvista_trueform").joinpath("_agents.md").read_text(
        encoding="utf-8")


__all__ = ["agents"]
