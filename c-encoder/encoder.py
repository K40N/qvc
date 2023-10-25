import numpy as np

from math import exp, log2, log10
import random

from box import Box

class ChunkEncoder:
    def __init__(self, xyt_volume_orig: np.array):
        # Here, (True, False) => (ok to place, not)
        self.xyt_volume = xyt_volume_orig.copy()

    def emit_box(self) -> Box:
        # Pick the best box we can
        ...
        # Update the xyt_volume property
        for x, y, t in best_box.member_coords():
            self.xyt_volume[x, y, t] = False
        # Return our chosen box
        return best_box
