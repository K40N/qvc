import numpy as np

from box import Box

class ChunkEncoder:
    def __init__(self, xyt_volume: np.array):
        self.xyt_volume = xyt_volume
        ...

    def emit_box(self) -> Box:
        return ...
