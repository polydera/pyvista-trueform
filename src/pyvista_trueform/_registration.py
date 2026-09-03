"""
Point-set registration and chamfer distance over PyVista datasets

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import trueform as tf

_METHODS = {
    "rigid": tf.fit_rigid_alignment,
    "similarity": tf.fit_similarity_alignment,
    "icp": tf.fit_icp_alignment,
    "obb": tf.fit_obb_alignment,
    "knn": tf.fit_knn_alignment,
}

_POINT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


def _point_cloud(source, name):
    if isinstance(source, tf.PointCloud):
        return source
    points = np.asarray(getattr(source, "points", source))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have (N, 3) points, got {points.shape}")
    if points.dtype not in _POINT_DTYPES:
        raise TypeError(
            f"{name} must have dtype float32 or float64, got {points.dtype}")
    return tf.PointCloud(np.ascontiguousarray(points))


def _matched_clouds(source, target):
    source = _point_cloud(source, "source")
    target = _point_cloud(target, "target")
    if source.dtype != target.dtype:
        raise TypeError(
            f"source and target must share one dtype, got {source.dtype} "
            f"and {target.dtype}")
    return source, target


def align(source, target, *, method="icp", **kwargs):
    """Fit the transformation carrying ``source`` onto ``target``.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects. The
    result is a homogeneous ``(4, 4)`` matrix mapping source world
    coordinates to target world coordinates — ready for
    ``dataset.transform(matrix)``.

    Methods (each forwards its keyword arguments to the trueform entry):

    - ``"rigid"`` — :func:`trueform.fit_rigid_alignment`; requires the two
      point sets in one-to-one correspondence (Kabsch), no kwargs.
    - ``"similarity"`` — :func:`trueform.fit_similarity_alignment`; rotation
      + uniform scale + translation, same correspondence requirement as
      ``"rigid"``, no kwargs.
    - ``"icp"`` — :func:`trueform.fit_icp_alignment`; iterative closest
      point with convergence detection (``max_iterations``, ``n_samples``,
      ``k``, ``sigma``, ``outlier_proportion``, ...).
    - ``"obb"`` — :func:`trueform.fit_obb_alignment`; oriented-bounding-box
      axes, no correspondences (``sample_size``).
    - ``"knn"`` — :func:`trueform.fit_knn_alignment`; one soft-correspondence
      step (``k``, ``sigma``, ``outlier_proportion``).

    The point-to-plane and normal-weighted variants take a
    ``(PointCloud, normals)`` tuple in trueform's own API and stay
    reachable by composition.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    if method not in _METHODS:
        supported = ", ".join(sorted(_METHODS))
        raise ValueError(
            f"unknown alignment method {method!r}; supported: {supported}")
    source, target = _matched_clouds(source, target)
    return _METHODS[method](source, target, **kwargs)


def chamfer_distance(source, target):
    """Mean nearest-neighbor distance from ``source`` to ``target``.

    One-way chamfer through :func:`trueform.chamfer_error`; average the two
    directions for the symmetric measure. Operands as in :func:`align`.

    Returns
    -------
    float
    """
    return tf.chamfer_error(*_matched_clouds(source, target))


__all__ = ["align", "chamfer_distance"]
