"""
Point-set registration and chamfer distance over PyVista datasets

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import numpy as np
import trueform as tf

from ._forward import _forwarded

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


def align_rigid(source, target):
    """Fit a rigid transformation (rotation + translation) onto ``target``.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects, in
    point-to-point correspondence (Kabsch). The result is the delta: a
    homogeneous ``(4, 4)`` matrix mapping ``source``'s current points
    onto ``target``, nothing of the source's own history composed in —
    apply it to the source itself to align,
    ``source.transform(matrix, inplace=False)``.

    No options: rigid alignment has none beyond the two clouds.

    The point-to-plane and normal-weighted variants take a
    ``(PointCloud, normals)`` tuple in trueform's own API and stay
    reachable by composition.

    See :func:`trueform.fit_rigid_alignment`.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    source, target = _matched_clouds(source, target)
    return tf.fit_rigid_alignment(source, target)


def align_similarity(source, target):
    """Fit a similarity transformation (rotation + uniform scale +
    translation) onto ``target``.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects, in
    point-to-point correspondence, the same requirement as
    :func:`align_rigid`. The result is the delta: a homogeneous
    ``(4, 4)`` matrix mapping ``source``'s current points onto
    ``target``, its linear part carrying the uniform scale, nothing of
    the source's own history composed in — apply it to the source
    itself to align, ``source.transform(matrix, inplace=False)``.

    No options: similarity alignment has none beyond the two clouds.

    See :func:`trueform.fit_similarity_alignment`.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    source, target = _matched_clouds(source, target)
    return tf.fit_similarity_alignment(source, target)


def align_icp(source, target, *, max_iterations=None, n_samples=None,
              k=None, sigma=None, outlier_proportion=None,
              min_relative_improvement=None, ema_alpha=None):
    """Fit a rigid transformation by Iterative Closest Point.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects; no
    correspondence is required. The result is the delta: a homogeneous
    ``(4, 4)`` matrix mapping ``source``'s current points onto
    ``target``, nothing of the source's own history composed in — apply
    it to the source itself to align,
    ``source.transform(matrix, inplace=False)``.

    Options (trueform's defaults apply when omitted):

    - ``max_iterations`` — maximum number of ICP iterations.
    - ``n_samples`` — points subsampled per iteration (0 = all).
    - ``k`` — number of nearest neighbors per correspondence.
    - ``sigma`` — Gaussian kernel width; adaptive when omitted.
    - ``outlier_proportion`` — proportion of worst correspondences rejected.
    - ``min_relative_improvement`` — convergence threshold.
    - ``ema_alpha`` — EMA smoothing factor for convergence detection.

    The point-to-plane and normal-weighted variants take a
    ``(PointCloud, normals)`` tuple in trueform's own API and stay
    reachable by composition.

    See :func:`trueform.fit_icp_alignment`.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    source, target = _matched_clouds(source, target)
    return tf.fit_icp_alignment(source, target, **_forwarded(
        max_iterations=max_iterations, n_samples=n_samples, k=k,
        sigma=sigma, outlier_proportion=outlier_proportion,
        min_relative_improvement=min_relative_improvement,
        ema_alpha=ema_alpha))


def align_obb(source, target, *, sample_size=None):
    """Fit a rigid transformation by aligning oriented bounding boxes.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects; no
    correspondence is used. The result is the delta: a homogeneous
    ``(4, 4)`` matrix mapping ``source``'s current points onto
    ``target``, nothing of the source's own history composed in — apply
    it to the source itself to align,
    ``source.transform(matrix, inplace=False)``.

    Options (trueform's default applies when omitted):

    - ``sample_size`` — points sampled to disambiguate the OBB's
      symmetry group.

    See :func:`trueform.fit_obb_alignment`.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    source, target = _matched_clouds(source, target)
    return tf.fit_obb_alignment(source, target, **_forwarded(
        sample_size=sample_size))


def align_knn(source, target, *, k=None, sigma=None, outlier_proportion=None):
    """Fit a rigid transformation from one soft k-nearest-neighbor
    correspondence step.

    ``source`` and ``target`` are any PyVista datasets with ``.points``,
    bare ``(N, 3)`` arrays, or :class:`trueform.PointCloud` objects; no
    correspondence is required. The result is the delta: a homogeneous
    ``(4, 4)`` matrix mapping ``source``'s current points onto
    ``target``, nothing of the source's own history composed in — apply
    it to the source itself to align,
    ``source.transform(matrix, inplace=False)``.

    Options (trueform's defaults apply when omitted):

    - ``k`` — number of nearest neighbors per correspondence.
    - ``sigma`` — Gaussian kernel width; the k-th neighbor's distance
      when omitted.
    - ``outlier_proportion`` — proportion of worst correspondences rejected.

    The point-to-plane and normal-weighted variants take a
    ``(PointCloud, normals)`` tuple in trueform's own API and stay
    reachable by composition.

    See :func:`trueform.fit_knn_alignment`.

    Returns
    -------
    np.ndarray of shape (4, 4)
    """
    source, target = _matched_clouds(source, target)
    return tf.fit_knn_alignment(source, target, **_forwarded(
        k=k, sigma=sigma, outlier_proportion=outlier_proportion))


def chamfer_distance(source, target):
    """Mean nearest-neighbor distance from ``source`` to ``target``.

    One-way chamfer through :func:`trueform.chamfer_error`; average the two
    directions for the symmetric measure. Operands as in :func:`align_rigid`.

    Returns
    -------
    float
    """
    return tf.chamfer_error(*_matched_clouds(source, target))


__all__ = ["align_rigid", "align_similarity", "align_icp", "align_obb",
           "align_knn", "chamfer_distance"]
