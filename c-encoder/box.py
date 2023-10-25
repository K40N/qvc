from dataclasses import dataclass
from math import log2
import typing
import sys

import numpy as np

from constants import XYT_BITS

MIN_SIZE = 0

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
            if cfg.var[0] > MIN_SIZE:
                opt = BoxCfg((cfg.var[0] - 1, cfg.var[1], cfg.var[2]))
                if opt not in seen:
                    yield opt
                    seen.add(opt)
            if cfg.var[1] > MIN_SIZE:
                opt = BoxCfg((cfg.var[0], cfg.var[1] - 1, cfg.var[2]))
                if opt not in seen:
                    yield opt
                    seen.add(opt)
            if cfg.var[2] > MIN_SIZE:
                opt = BoxCfg((cfg.var[0], cfg.var[1], cfg.var[2] - 1))
                if opt not in seen:
                    yield opt
                    seen.add(opt)

    @classmethod
    def find_best_box(cls, xyt_ok: np.array, p_threshold: float) -> typing.Union["Box", None]:
        cfgs = [ cls(XYT_BITS) ]
        best_box, best_box_count = None, 0
        while (best_box == None) and (len(cfgs) > 0):
            cfgs = list(BoxCfg.subdivided(cfgs))
            for cfg in cfgs:
                for box in cfg.possible_boxes():
                    sys.stdout.flush()
                    count, vol = 0, 0
                    for x, y, t in box.member_coords():
                        if xyt_ok[x,y,t]:
                            count += 1
                        vol += 1
                    if (count / vol) > p_threshold:
                        if count > best_box_count:
                            best_box = box
                            count = best_box_count
        return best_box

@dataclass(frozen=True)
class Box:
    xyt_fixed: tuple[int, int, int]
    box_cfg: BoxCfg

    def member_coords(self) -> typing.Generator[tuple[int, int, int], None, None]:
        for x_offset in range(1 << self.box_cfg.var[0]):
            for y_offset in range(1 << self.box_cfg.var[1]):
                for t_offset in range(1 << self.box_cfg.var[2]):
                    x = x_offset + (self.xyt_fixed[0] << self.box_cfg.var[0])
                    y = y_offset + (self.xyt_fixed[1] << self.box_cfg.var[1])
                    t = t_offset + (self.xyt_fixed[2] << self.box_cfg.var[2])
                    yield x, y, t

@dataclass(frozen=True)
class EncodedChunk:
    boxes: list[Box | None]
