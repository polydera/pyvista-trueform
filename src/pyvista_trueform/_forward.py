"""
Selective keyword forwarding onto trueform callees

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""


def _forwarded(**kwargs):
    """The entries of ``kwargs`` whose value is not ``None``.

    Every public entry mirrors its trueform callee's options as explicit,
    keyword-only, ``None``-defaulted parameters, so trueform stays the sole
    producer of every default. This collects the ones the caller actually
    set, for a call site to forward with ``**_forwarded(...)``; the omitted
    ones fall through to the callee's own default.
    """
    return {key: value for key, value in kwargs.items() if value is not None}


__all__ = []
