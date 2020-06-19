import numpy as np

SOLID, DASH = 0, 1
R_STEP = 0
N_STATES = 7
SINGLE_STATE = 7


class BairdMDP:
  def __init__(self):
    self.get_moves()
    self.get_states()
    self.reset()

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
