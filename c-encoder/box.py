from dataclasses import dataclass
from math import log2
import typing

import numpy as np

from constants import XYT_BITS

@dataclass(frozen=True)
class BoxCfg:
    var: tuple[int, int, int]

    def volume(self) -> int:
        return (1 << self.var[0]) * (1 << self.var[1]) * (1 << self.var[2])

    def possible_boxes(self) -> typing.Generator["Box", None, None]:
        for fixed_x in range(0, 1 << (XYT_BITS[0] - self.var[0])):
            for fixed_y in range(0, 1 << (XYT_BITS[1] - self.var[1])):
                for fixed_t in range(0, 1 << (XYT_BITS[2] - self.var[2])):
                    yield Box(
                        (fixed_x, fixed_y, fixed_t),
                        self,
                    )

    @classmethod
    def subdivided(cls, box_cfgs: list["BoxCfg"]) -> typing.Generator["BoxCfg", None, None]:
        seen = set()
        for cfg in box_cfgs:
            if (cfg.var[0] % 2) == 0:
                opt = BoxCfg((cfg.var[0] // 2, cfg.var[1], cfg.var[2]))
                if opt not in seen:
                    yield opt
                    seen.add(opt)
            if (cfg.var[1] % 2) == 0:
                opt = BoxCfg((cfg.var[0], cfg.var[1] // 2, cfg.var[2]))
                if opt not in seen:
                    yield opt
                    seen.add(opt)
            if (cfg.var[2] % 2) == 0:
                opt = BoxCfg((cfg.var[0], cfg.var[1], cfg.var[2] // 2))
                if opt not in seen:
                    yield opt
                    seen.add(opt)

    @classmethod
    def find_best_box(self, xyt_ok: np.array, p_threshold: float) -> Box:
        cfgs = [ BoxCfg(XYT_BITS) ]
        best_box, best_box_count = None, 0
        while best_box == None:
            for cfg in BoxCfg.subdivided(cfgs):
                for box in cfg.possible_boxes():

        return best_box

@dataclass(frozen=True)
class Box:
    xyt_fixed: tuple[int, int, int]
    box_cfg: BoxCfg

@dataclass(frozen=True)
class EncodedChunk:
    boxes: list[Box]
