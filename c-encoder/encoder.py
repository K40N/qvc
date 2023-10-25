import numpy as np

from math import exp, log2, log10
import random
import sys

from box import Box, BoxCfg

class ChunkEncoder:
    def __init__(self, xyt_volume_orig: np.array):
        # Here, (True, False) => (ok to place, not)
        self.xyt_volume = xyt_volume_orig.copy()

    def emit_box(self) -> Box | None:
        # Pick the best box we can
        best_box = BoxCfg.find_best_box(self.xyt_volume, 0.9)
        if best_box is None:
            print("/!\\", end="")
            sys.stdout.flush()
            return None
        # Update the xyt_volume property
        for x, y, t in best_box.member_coords():
            self.xyt_volume[x, y, t] = False
        # Return our chosen box
        return best_box
    