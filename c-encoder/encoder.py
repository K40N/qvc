import numpy as np

from math import exp

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
            if self.sa_p_function(e_now, e_canidate, i):
                state, e_now = state_canidate, e_canidate
            if e_now > e_best:
                best_state, e_best = state, e_now
            iteration += 1
            temp = self.sa_temperature(iteration)
        # Update the xyt_volume property
        for x, y, t in chosen_box.member_coords():
            self.xyt_volume[x, y, t] = False
        # Return our chosen box
        return chosen_box

    def sa_initial_state(self) -> Box:
        return ...
    
    def sa_neighbor_of(self, state: Box) -> Box:
        return ...
    
    def sa_energy(self, state: Box) -> Box:
        return ...

    def sa_p_function(self, e_now: float, e_canidate: float, temp: int) -> float:
        delta_e = e_now - e_canidate
        unbounded_result = exp(-delta_e / temp)
        return min(unbounded_result, 1.0)

    def sa_temperature(self, iteration: int): float:
        return SA_TEMP_MAX - (iteration * SA_TEMP_ALPHA)
    
    SA_TEMP_MAX = 10.0
    SA_TEMP_ALPHA = 0.2
