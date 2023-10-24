from dataclasses import dataclass

@dataclass(frozen=True)
class Box:
    x:      int
    y:      int
    t:      int
    log2_w: int # w = width (X)
    log2_h: int # h = height (Y)
    log2_d: int # d = duration (t)
    
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
