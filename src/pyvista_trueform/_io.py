"""
Mesh and volume file reading and writing through trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

from pathlib import Path

import trueform as tf

from ._conversion import (to_pyvista, to_trueform, volume_to_pyvista,
                          volume_to_trueform)
from ._forward import _forwarded

# Each entry is the trueform call and the conversion for its carrier:
# meshes cross through the mesh pair, volumes through the volume pair.
_READERS = {
    ".stl": (tf.read_stl, to_pyvista),
    ".obj": (tf.read_obj, to_pyvista),
    ".nii": (tf.read_nifti, volume_to_pyvista),
    ".nii.gz": (tf.read_nifti, volume_to_pyvista),
}
_WRITERS = {
    ".stl": (tf.write_stl, to_trueform),
    ".obj": (tf.write_obj, to_trueform),
    ".nii": (tf.write_nifti, volume_to_trueform),
    ".nii.gz": (tf.write_nifti, volume_to_trueform),
}


def _suffix(path):
    """The dispatch suffix of a path, ``.nii.gz`` counted whole."""
    name = Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(name).suffix


def _dispatched(path, table):
    suffix = _suffix(path)
    if suffix not in table:
        supported = ", ".join(sorted(table))
        raise ValueError(
            f"unsupported file suffix {suffix!r} for {path!r}; "
            f"supported: {supported}")
    return table[suffix]


def read(path, *, index_dtype=None, ngon=None, dtype=None):
    """Read a mesh or volume file into a fresh PyVista dataset.

    Dispatches on the path suffix: ``.stl`` through
    :func:`trueform.read_stl` (parallel, duplicate vertices welded),
    ``.obj`` through :func:`trueform.read_obj` (polygon sizes preserved),
    ``.nii`` and ``.nii.gz`` through :func:`trueform.read_nifti` (NIfTI-1,
    native dtype). A mesh file answers a
    :class:`pyvista.PolyData` through :func:`pyvista_trueform.to_pyvista`,
    a volume file an :class:`pyvista.ImageData` through
    :func:`pyvista_trueform.volume_to_pyvista`; both conversions are
    zero-copy.

    A posed scan lands the file's affine on the grid, so a CT reads back
    with its patient-space direction and origin already in place, and its
    isosurface is a patient-space mesh with no further step.
    ``trueform.read_nifti_header(path)`` answers the file's facts (dtype,
    dims, spacing, units, whether it is posed) without reading its
    samples.

    Parameters
    ----------
    path : str or os.PathLike
        File to read.
    index_dtype : dtype, optional
        Forwarded to the trueform mesh readers. Trueform's default
        applies when omitted.
    ngon : optional
        Forwarded to :func:`trueform.read_obj` only.
    dtype : optional
        Forwarded to :func:`trueform.read_obj` (point dtype) and to
        :func:`trueform.read_nifti` (sample dtype); ``.stl`` has no such
        option, so passing one for an ``.stl`` path raises there.
        Trueform's defaults apply when omitted.

    Returns
    -------
    pyvista.PolyData or pyvista.ImageData
    """
    reader, converted = _dispatched(path, _READERS)
    return converted(reader(str(path), **_forwarded(
        index_dtype=index_dtype, ngon=ngon, dtype=dtype)))


def write(path, dataset, *, transformation=None):
    """Write a PyVista dataset to a mesh or volume file through trueform.

    Dispatches on the path suffix: ``.stl`` through
    :func:`trueform.write_stl` (triangles only), ``.obj`` through
    :func:`trueform.write_obj` (any polygon sizes), ``.nii`` and
    ``.nii.gz`` through :func:`trueform.write_nifti` (NIfTI-1, gzipped by
    the extension). A mesh path takes a polygon-only
    :class:`pyvista.PolyData` through
    :func:`pyvista_trueform.to_trueform`, a volume path an
    :class:`pyvista.ImageData` through
    :func:`pyvista_trueform.volume_to_trueform` — its active point
    scalars are the field written, and its placement is composed into the
    file's affine.

    Parameters
    ----------
    path : str or os.PathLike
        Output file path.
    dataset : pyvista.PolyData or pyvista.ImageData
        The mesh or volume, matching the path's carrier.
    transformation : ndarray, optional
        Forwarded to the trueform mesh writers; overrides any
        transformation set on a :class:`trueform.Mesh` operand. A volume
        carries its own placement, so :func:`trueform.write_nifti` takes
        no such option and passing one for a ``.nii`` path raises there.
        Trueform's default applies when omitted.
    """
    writer, converted = _dispatched(path, _WRITERS)
    if not writer(converted(dataset), str(path),
                  **_forwarded(transformation=transformation)):
        raise OSError(f"trueform failed to write {path!r}")


__all__ = ["read", "write"]
