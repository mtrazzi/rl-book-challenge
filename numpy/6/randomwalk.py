import numpy as np

N_STATES = 5
EMPTY_MOVE = 0
STATES = np.arange(N_STATES)
P_LEFT = 0.5
R_STEP = 0

class RandomWalk:
  def __init__(self):
    self.reset()

  @property
  def moves(self):
    return [EMPTY_MOVE]

  @property
  def states(self):
    return STATES

  def sample_shift(self):
    return np.sign(np.random.random() - P_LEFT)

  def step(self, action):
    new_state = self.state + self.sample_shift() 
    if not (0 <= new_state < N_STATES):
      return self.state, new_state == N_STATES, True, {}
    self.state = new_state
    return self.state, R_STEP, False, {}

  def reset(self):
    self.state = N_STATES // 2
    return self.state

  def __str__(self):
    return self.state
