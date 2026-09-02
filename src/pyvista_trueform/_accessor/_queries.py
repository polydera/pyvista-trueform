"""
Diagnostics and spatial queries on the .trueform accessor

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import trueform as tf


class _QueriesMixin:

    def is_closed(self):
        """True when every edge is shared by exactly two faces (watertight)."""
        return tf.is_closed(self.to_mesh())

    def is_manifold(self):
        """True when no edge is shared by more than two faces."""
        return tf.is_manifold(self.to_mesh())
