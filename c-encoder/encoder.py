import numpy as np

from math import exp, log2, log10
import random

from box import Box

# Break the image into boxes using simulated annealing
class ChunkEncoder:
    def __init__(self, xyt_volume_orig: np.array):
        # Here, (True, False) => (ok to place, not)
        self.xyt_volume = xyt_volume_orig.copy()

    def emit_box(self) -> Box:
        # Perform simulated annealing to find a place
        # to put the box
        state = self.sa_initial_state()
        e_now = self.sa_energy(state)
        best_state, e_best = state, e_now
        iteration = 0
        temp = self.sa_temperature(iteration)
        while temp > 0.0:
            state_canidate = self.sa_neighbor_of(state)
            e_canidate = self.sa_energy(state_canidate)
            if self.sa_p_function(e_now, e_canidate, temp):
                state, e_now = state_canidate, e_canidate
            if e_now < e_best:
                best_state, e_best = state, e_now
            iteration += 1
            temp = self.sa_temperature(iteration)
        # Update the xyt_volume property
        for x, y, t in best_state.member_coords():
            self.xyt_volume[x, y, t] = False
        # Return our chosen box
        return best_state

    def sa_initial_state(self) -> Box:
        x_size, y_size, t_size = self.xyt_volume.shape
        cx, cy, ct = (x_size // 2), (y_size // 2), (t_size // 2)
        qx, qy, qt = (x_size // 4), (y_size // 4), (t_size // 4)
        return Box(
            cx - qx, cy - qy, ct - qt,
            int(log2(cx)), int(log2(cy)), int(log2(ct)),
        )
    
    def sa_neighbor_of(self, state: Box) -> Box:
        change = [random.choice([-1, 1]), *(0 for _ in range(5))]
        random.shuffle(change)
        box = Box(
            state.x + change[0], state.y + change[1], state.t + change[2],
            state.log2_w + change[3], state.log2_h + change[4], state.log2_d + change[5],
        )
        return box.ensure_within(self.xyt_volume.shape)
    
    def sa_energy(self, state: Box) -> Box:
        count, volume = 0, 0
        for x, y, t in state.member_coords():
            if self.xyt_volume[x, y, t]:
                count += 1
            volume += 1
        x_size, y_size, t_size = self.xyt_volume.shape
        max_volume = x_size * y_size * t_size
        proportion_within = count / volume
        proportion_volume = volume / max_volume
        assert proportion_within <= 1.0
        assert proportion_volume <= 1.0
        proportion_combined = (proportion_within + proportion_volume) / 2
        return 1.0 - proportion_combined

    def sa_p_function(self, e_now: float, e_canidate: float, temp: int) -> float:
        delta_e = e_now - e_canidate
        unbounded_result = exp(-delta_e / temp)
        return min(unbounded_result, 1.0)

    def sa_temperature(self, iteration: int) -> float:
        return self.SA_TEMP_MAX - (iteration * self.SA_TEMP_ALPHA)
    
    SA_TEMP_MAX = 20.0
    SA_TEMP_ALPHA = 0.01
