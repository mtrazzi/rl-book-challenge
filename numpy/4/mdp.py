from abc import ABC, abstractmethod
import time
import numpy as np


class MDP(ABC):
    """Base environment for the dynamic programming chapter."""

    def __init__(self):
        self.init_p()

    def renormalize(self):
        for s in self.states:
            for a in self.moves:
                p_sum = sum([self.p[(s_p, r, s, a)] for s_p in self.states
                             for r in self.r])
                if p_sum > 0:
                    for s_p in self.states:
                        for r in self.r:
                            self.p[(s_p, r, s, a)] /= p_sum

    def init_p(self):
        print("starting to compute transitions p...")
        start = time.time()
        self.p = {(s_p, r, s, a): self._p(s_p, r, s, a)
                  for s in self.states for a in self.moves
                  for s_p in self.states for r in self.r}
        # hardcoded normalization to avoid overflow
        self.renormalize()

        def p_sum(s_p_list, r_list, s_list, a_list):
            return np.sum([self.p[(s_p, r, s, a)] for s_p in s_p_list
                           for r in r_list for s in s_list for a in a_list])
        self.pr = {(s, a): np.array([p_sum(self.states, [r], [s], [a])
                   for r in self.r]) for s in self.states for a in self.moves}
        self.psp = {(s, a): np.array([p_sum([s_p], self.r, [s], [a])
                    for s_p in self.states])
                    for s in self.states for a in self.moves}
        print(f"finished after {time.time()-start}s")
