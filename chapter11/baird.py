import numpy as np

SOLID, DASH = 0, 1
R_STEP = 0
N_STATES = 7
SINGLE_STATE = 7


class BairdMDP:
  def __init__(self):
    self.get_moves()
    self.get_states()
    self.get_rewards()
    self.init_p()
    self.reset()

  def init_p(self):
    self.p = {(s_p, r, s, a): self._p(s_p, r, s, a)
              for s in self.states for a in self.moves
              for s_p in self.states for r in self.r}

  def get_rewards(self):
    self.r = [R_STEP]

  def get_moves(self):
    self.moves = [SOLID, DASH]

  def get_states(self):
    self.states = list(range(1, N_STATES + 1))

  def step(self, a):
    self.state = SINGLE_STATE if a == SOLID else np.random.randint(1, N_STATES)
    return self.state, R_STEP, False, {}

  def seed(self, seed):
    np.random.seed(seed)

  def reset(self):
    self.state = SINGLE_STATE
    return self.state

  def _p(self, s_p, r, s, a):
    return a == SOLID if s_p == SINGLE_STATE else (a == DASH) / 6
