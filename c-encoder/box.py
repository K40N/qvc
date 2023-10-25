from dataclasses import dataclass
from math import log2
import typing

@dataclass(frozen=True)
class Box:
    x:      int
    y:      int
    t:      int
    log2_w: int # w = width (X)
    log2_h: int # h = height (Y)
    log2_d: int # d = duration (t)

    def ensure_within(self, shape: tuple[int, int, int]) -> "Box":
        x_size, y_size, t_size = shape
        x_new, y_new, t_new = (
            min(max(self.x, 0), x_size - 1),
            min(max(self.y, 0), y_size - 1),
            min(max(self.t, 0), t_size - 1),
        )
        return Box(
            x_new, y_new, t_new,
            min(max(self.log2_w, 0), int(log2(x_size - x_new))),
            min(max(self.log2_h, 0), int(log2(y_size - y_new))),
            min(max(self.log2_d, 0), int(log2(t_size - t_new))),
        )

    def member_coords(self) -> typing.Generator[tuple[int, int, int], None, None]:
        for x_ in self.offset_by_bits(self.x, self.log2_w):
            for y_ in self.offset_by_bits(self.y, self.log2_h):
                for t_ in self.offset_by_bits(self.t, self.log2_d):
                    yield x_, y_, t_
    
    def offset_by_bits(self, x: int, log2_w: int) -> typing.Generator[int, None, None]:
        for i in range(1 << log2_w):
            yield x + i
    
    def volume(self) -> int:
        return (2**self.log2_w) + (2**self.log2_h) + (2**self.log2_d)
    
    def contains_point(self, qx: int, qy: int, qt: int):
        x_ok = ((qx - self.x) >= 0) and ((qx - self.x) <= (1 << self.log2_w))
        y_ok = ((qy - self.y) >= 0) and ((qy - self.y) <= (1 << self.log2_h))
        t_ok = ((qt - self.t) >= 0) and ((qt - self.t) <= (1 << self.log2_d))
        return x_ok and y_ok and t_ok

def box_union_contains_point(box_union_of: list[Box], xyt: (int, int, int)):
    x, y, t = xyt
    for box in box_union_of:
        if box.contains_point(x, y, t):
            return True
    return False

@dataclass(frozen=True)
class EncodedChunk:
    boxes: list[Box]
