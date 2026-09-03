"""
Mesh file reading and writing through trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

from pathlib import Path

import trueform as tf

from ._conversion import to_pyvista, to_trueform
from ._forward import _forwarded

_READERS = {".stl": tf.read_stl, ".obj": tf.read_obj}
_WRITERS = {".stl": tf.write_stl, ".obj": tf.write_obj}


def _dispatched(path, table):
    suffix = Path(path).suffix.lower()
    if suffix not in table:
        supported = ", ".join(sorted(table))
        raise ValueError(
            f"unsupported mesh file suffix {suffix!r} for {path!r}; "
            f"supported: {supported}")
    return table[suffix]


def read(path, *, index_dtype=None, ngon=None, dtype=None):
    """Read a mesh file into a fresh PyVista PolyData through trueform.

    Dispatches on the path suffix: ``.stl`` through
    :func:`trueform.read_stl` (parallel, duplicate vertices welded),
    ``.obj`` through :func:`trueform.read_obj` (polygon sizes preserved).
    The arrays convert through :func:`pyvista_trueform.to_pyvista`
    zero-copy.

    Parameters
    ----------
    path : str or os.PathLike
        Mesh file to read.
    index_dtype : dtype, optional
        Forwarded to the trueform reader (both formats). Trueform's
        default applies when omitted.
    ngon, dtype : optional
        Forwarded to :func:`trueform.read_obj` only — ``.stl`` has no such
        options, so passing either for an ``.stl`` path raises there.
        Trueform's defaults apply when omitted.

    Returns
    -------
    pyvista.PolyData
    """
    reader = _dispatched(path, _READERS)
    return to_pyvista(reader(str(path), **_forwarded(
        index_dtype=index_dtype, ngon=ngon, dtype=dtype)))


def write(path, dataset, *, transformation=None):
    """Write a polygonal PyVista dataset to a mesh file through trueform.

    Dispatches on the path suffix: ``.stl`` through
    :func:`trueform.write_stl` (triangles only), ``.obj`` through
    :func:`trueform.write_obj` (any polygon sizes). The dataset converts
    through :func:`pyvista_trueform.to_trueform`.

    Parameters
    ----------
    path : str or os.PathLike
        Output file path.
    dataset : pyvista.PolyData
        Polygon-only dataset (no vertices, lines, or strips).
    transformation : ndarray, optional
        Forwarded to the trueform writer (both formats); overrides any
        transformation set on a :class:`trueform.Mesh` operand. Trueform's
        default applies when omitted.
    """
    writer = _dispatched(path, _WRITERS)
    if not writer(to_trueform(dataset), str(path),
                  **_forwarded(transformation=transformation)):
        raise OSError(f"trueform failed to write {path!r}")


__all__ = ["read", "write"]
