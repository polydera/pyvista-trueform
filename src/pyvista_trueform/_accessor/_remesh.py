"""
Remeshing on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf

from .._forward import _forwarded


class _RemeshMixin:

    def _remeshed(self, result):
        if len(result) == 3:
            faces, points, labels = result
            return self._labeled((faces, points), labels)
        return self._labeled(result)

    def remeshed(self, target_length, *, iterations=None,
                 relaxation_iters=None, min_quality=None, lambda_=None,
                 preserve_boundary=None, use_quadric=None, parallel=None,
                 feature_angle=None, feature_weight=None,
                 preserve_regions=None):
        """Isotropic remesh toward ``target_length`` edges.

        Parallel split/collapse/flip/relax; boundary-, feature- and
        region-aware. With ``preserve_regions=`` (one label per input face)
        the output labels ride as ``trueform_labels``. When omitted,
        trueform's defaults apply — see :func:`trueform.isotropic_remeshed`
        for what each option controls.
        """
        return self._remeshed(tf.isotropic_remeshed(
            self.to_mesh(), target_length,
            **_forwarded(iterations=iterations,
                        relaxation_iters=relaxation_iters,
                        min_quality=min_quality, lambda_=lambda_,
                        preserve_boundary=preserve_boundary,
                        use_quadric=use_quadric, parallel=parallel,
                        feature_angle=feature_angle,
                        feature_weight=feature_weight,
                        preserve_regions=preserve_regions)))

    def decimated(self, target_proportion, *, min_quality=None,
                 preserve_boundary=None, stabilizer=None, parallel=None,
                 feature_angle=None, feature_weight=None,
                 preserve_regions=None):
        """Quadric-error decimation to ``target_proportion`` of the faces.

        With ``preserve_regions=`` the output labels ride as
        ``trueform_labels``. When omitted, trueform's defaults apply — see
        :func:`trueform.decimated` for what each option controls.
        """
        return self._remeshed(tf.decimated(
            self.to_mesh(), target_proportion,
            **_forwarded(min_quality=min_quality,
                        preserve_boundary=preserve_boundary,
                        stabilizer=stabilizer, parallel=parallel,
                        feature_angle=feature_angle,
                        feature_weight=feature_weight,
                        preserve_regions=preserve_regions)))

    def simplified(self, *, error_rel=None, optimize_iterations=None,
                  iterations=None, relaxation_iters=None, lambda_=None,
                  min_quality=None, preserve_boundary=None, stabilizer=None,
                  parallel=None, feature_angle=None, feature_weight=None,
                  preserve_regions=None):
        """Quadric-error simplification to an error budget.

        Flat regions collapse for almost no error while curved detail and
        feature edges survive; there is no target face count. With
        ``preserve_regions=`` the output labels ride as
        ``trueform_labels``. When omitted, trueform's defaults apply — see
        :func:`trueform.simplified` for what each option controls.
        """
        return self._remeshed(tf.simplified(
            self.to_mesh(),
            **_forwarded(error_rel=error_rel,
                        optimize_iterations=optimize_iterations,
                        iterations=iterations,
                        relaxation_iters=relaxation_iters, lambda_=lambda_,
                        min_quality=min_quality,
                        preserve_boundary=preserve_boundary,
                        stabilizer=stabilizer, parallel=parallel,
                        feature_angle=feature_angle,
                        feature_weight=feature_weight,
                        preserve_regions=preserve_regions)))
