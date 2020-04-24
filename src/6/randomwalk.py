import numpy as np

N_STATES = 5
EMPTY_MOVE = 0
STATES = list(range(N_STATES))
P_LEFT = 0.5
P_LEFT_L = [0.9, 0.1]
R_STEP = 0
ABSORBING_STATE = N_STATES
LEFT = 0
RIGHT = 1

class RandomWalk:
  def __init__(self):
    self.reset()

  @property
  def moves(self):
    return [EMPTY_MOVE]

  @property
  def states(self):
    return STATES + [ABSORBING_STATE]

  def sample_shift(self):
    return np.sign(np.random.random() - P_LEFT)

  def step(self, action):
    new_state = self.state + self.sample_shift() 
    if not (0 <= new_state < N_STATES):
      return ABSORBING_STATE, float(new_state == N_STATES), True, {}
    self.state = new_state
    return self.state, R_STEP, False, {}

  def reset(self):
    self.state = N_STATES // 2
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def __str__(self):
    return self.state

class NotSoRandomWalk:
  def __init__(self):
    self.reset()

  @property
  def moves(self):
    return [LEFT, RIGHT]

  @property
  def states(self):
    return STATES + [ABSORBING_STATE]

  def sample_shift(self, action):
    return np.sign(np.random.random() - P_LEFT_L[action])

  def step(self, action):
    new_state = self.state + self.sample_shift(action) 
    if not (0 <= new_state < N_STATES):
      return ABSORBING_STATE, float(new_state == N_STATES), True, {}
    self.state = new_state
    return self.state, R_STEP, False, {}

  def reset(self):
    self.state = N_STATES // 2
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def __str__(self):
    return self.state
