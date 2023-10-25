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
        best_box = self.pick_optimal_box()
        # Update the xyt_volume property
        for x, y, t in best_box.member_coords():
            self.xyt_volume[x, y, t] = False
        # Return our chosen box
        return best_box
    
    def pick_optimal_box(self) -> Box:
        best_box, best_vol = None, 0.0
        x_size, y_size, t_size = self.xyt_volume.shape
        for x in range(x_size):
            for y in range(y_size):
                for t in range(t_size):
                    if self.xyt_volume[x,y,t]:
                        # Dim: X
                        log2_px, done = 0, False
                        while not done:
                            try:
                                pos = self.xyt_volume[x + (1 << log2_px),y,t]
                                neg = self.xyt_volume[x - (1 << log2_px),y,t]
                            except IndexError:
                                log2_px = max(0, log2_px - 1)
                                done = True
                                break
                            if pos and neg:
                                log2_px += 1
                            else:
                                log2_px = max(0, log2_px - 1)
                                done = True
                        # Dim: Y
                        log2_py, done = 0, False
                        while not done:
                            try:
                                pos = self.xyt_volume[x,y + (1 << log2_py),t]
                                neg = self.xyt_volume[x,y - (1 << log2_py),t]
                            except IndexError:
                                log2_py = max(0, log2_py - 1)
                                done = True
                                break
                            if pos and neg:
                                log2_py += 1
                            else:
                                log2_py = max(0, log2_py - 1)
                                done = True
                        # Dim: T
                        log2_pt, done = 0, False
                        while not done:
                            try:
                                pos = self.xyt_volume[x,y,t + (1 << log2_pt)]
                                neg = self.xyt_volume[x,y,t - (1 << log2_pt)]
                            except IndexError:
                                log2_pt = max(0, log2_pt - 1)
                                done = True
                                break
                            if pos and neg:
                                log2_pt += 1
                            else:
                                log2_pt = max(0, log2_pt - 1)
                                done = True
                        # Check if best
                        canidate_box = Box(
                            x, y, t, log2_px, log2_py, log2_pt,
                        )
                        canidate_vol = canidate_box.volume()
                        if canidate_vol > best_vol:
                            best_box = canidate_box
                            best_vol = canidate_vol
        return best_box
