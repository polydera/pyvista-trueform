"""
Remeshing on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf


class _RemeshMixin:

    def _remeshed(self, result):
        if len(result) == 3:
            faces, points, labels = result
            return self._labeled((faces, points), labels)
        return self._labeled(result)

    def remeshed(self, target_length, **kwargs):
        """Isotropic remesh toward ``target_length`` edges.

        Parallel split/collapse/flip/relax; boundary-, feature- and
        region-aware. With ``preserve_regions=`` (one label per input face)
        the output labels ride as ``trueform_labels``. See
        :func:`trueform.isotropic_remeshed` for keyword arguments.
        """
        return self._remeshed(
            tf.isotropic_remeshed(self.to_mesh(), target_length, **kwargs))

    def decimated(self, target_proportion, **kwargs):
        """Quadric-error decimation to ``target_proportion`` of the faces.

        With ``preserve_regions=`` the output labels ride as
        ``trueform_labels``. See :func:`trueform.decimated` for keyword
        arguments.
        """
        return self._remeshed(
            tf.decimated(self.to_mesh(), target_proportion, **kwargs))

    def simplified(self, **kwargs):
        """Quadric-error simplification to an error budget.

        Flat regions collapse for almost no error while curved detail and
        feature edges survive; there is no target face count. With
        ``preserve_regions=`` the output labels ride as
        ``trueform_labels``. See :func:`trueform.simplified` for keyword
        arguments.
        """
        return self._remeshed(tf.simplified(self.to_mesh(), **kwargs))
