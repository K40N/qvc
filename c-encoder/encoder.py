import numpy as np

from box import Box

class ChunkEncoder:
    def __init__(self, xyt_volume_orig: np.array):
        # Here, (True, False) => (ok to place, not)
        self.xyt_volume = xyt_volume_orig.copy()

    def emit_box(self) -> Box:
        return ...
