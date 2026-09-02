"""
Ray picking and proximity that name the block

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

from typing import NamedTuple

import numpy as np
import pyvista as pv
import trueform as tf


class RayHit(NamedTuple):
    """The nearest ray hit across a target's blocks, from :func:`pick`."""

    block_index: int
    """Flat index of the hit block; ``0`` for a plain PolyData target."""
    block: pv.PolyData
    """The hit block itself; the dataset for a plain PolyData target."""
    face: int
    """The hit face on ``block``."""
    point: np.ndarray
    """The hit point, ``origin + t * direction``."""
    t: float
    """The ray parameter of the hit."""


class ClosestHit(NamedTuple):
    """The nearest block to a query, from :func:`closest`."""

    block_index: int
    """Flat index of the winning block; ``0`` for a plain PolyData target."""
    block: pv.PolyData
    """The winning block itself; the dataset for a plain PolyData target."""
    face: int
    """The nearest face on ``block``."""
    point: np.ndarray
    """The witness point on ``block``."""
    distance: float
    """The euclidean distance between the witness points."""


def _leaves(multiblock):
    for block in multiblock:
        if isinstance(block, pv.MultiBlock):
            yield from _leaves(block)
        else:
            yield block


def _flat_blocks(target):
    """Yield ``(block_index, block)`` over the target's flat leaf slots.

    A plain PolyData is its own single block at index 0. A MultiBlock
    flattens depth-first; every leaf slot keeps its flat number, ``None``
    slots are skipped, and a non-``None`` leaf that is not a PolyData
    raises.
    """
    if isinstance(target, pv.PolyData):
        yield 0, target
        return
    if not isinstance(target, pv.MultiBlock):
        raise TypeError(
            "target must be a pyvista.PolyData or pyvista.MultiBlock, "
            f"got {type(target).__name__}")
    for index, block in enumerate(_leaves(target)):
        if block is None:
            continue
        if not isinstance(block, pv.PolyData):
            raise TypeError(
                f"block {index} must be a pyvista.PolyData or None, "
                f"got {type(block).__name__}")
        yield index, block


def pick(target, ray):
    """The first face of ``target`` in the ray's way, naming its block.

    Every block answers through its own accessor cache
    (:meth:`~pyvista_trueform.TrueformAccessor.ray_cast`), so the spatial
    trees amortize across picks; the smallest ray parameter wins.

    Parameters
    ----------
    target : pyvista.PolyData or pyvista.MultiBlock
        The scene: one dataset, or blocks of them — any MultiBlock, e.g.
        the domains of a :func:`pyvista_trueform.csg_graph`. Nested
        MultiBlocks flatten; ``None`` blocks are skipped.
    ray : trueform.Ray
        A single ray; picking a batch of rays is future work.

    Returns
    -------
    RayHit or None
        ``None`` when every block misses.
    """
    if not isinstance(ray, tf.Ray):
        raise TypeError(
            f"ray must be a trueform.Ray, got {type(ray).__name__}")
    if ray.is_batch:
        raise ValueError(
            f"pick takes a single ray, got a batch of {ray.count}; "
            "batch picking is future work")
    best = None
    for index, block in _flat_blocks(target):
        hit = block.trueform.ray_cast(ray)
        if hit is None:
            continue
        face, t = hit
        if best is None or t < best[3]:
            best = (index, block, face, t)
    if best is None:
        return None
    index, block, face, t = best
    point = (np.asarray(ray.origin, dtype=np.float64)
             + t * np.asarray(ray.direction, dtype=np.float64))
    return RayHit(index, block, face, point, t)


def closest(target, query):
    """The block of ``target`` nearest to ``query``, with its witness.

    A length-3 ``query`` asks each block through its accessor's
    :meth:`~pyvista_trueform.TrueformAccessor.closest_point`; a dataset or
    :class:`trueform.Mesh` through its
    :meth:`~pyvista_trueform.TrueformAccessor.closest_point_pair` (faced
    operands mesh-to-mesh, faceless ones as a batched point query). The
    smallest distance wins, and the returned ``point`` is the witness ON
    the winning block.

    Parameters
    ----------
    target : pyvista.PolyData or pyvista.MultiBlock
        The scene, exactly as in :func:`pick`.
    query : array-like of shape (3,), pyvista dataset, or trueform.Mesh
        A point, or any operand
        :meth:`~pyvista_trueform.TrueformAccessor.closest_point_pair`
        takes.

    Returns
    -------
    ClosestHit or None
        ``None`` when the target has no blocks.
    """
    query_is_dataset = isinstance(
        query, (tf.Mesh, pv.PolyData, pv.PointSet))
    best = None
    for index, block in _flat_blocks(target):
        if query_is_dataset:
            (face, _), (distance, point, _) = \
                block.trueform.closest_point_pair(query)
        else:
            face, distance, point = block.trueform.closest_point(query)
        if best is None or distance < best.distance:
            best = ClosestHit(index, block, face, point, distance)
    return best


__all__ = ["ClosestHit", "RayHit", "closest", "pick"]
