import numpy as np

N_STATES = 19
EMPTY_MOVE = 0
P_LEFT = 0.5
P_LEFT_L = [0.9, 0.1]
R_STEP = 0
ABSORBING_STATE = N_STATES
LEFT = 0
RIGHT = 1
R_RIGHT = 1

class RandomWalk:
  def __init__(self, n_states=None, r_l=-1):
    self.n_states = N_STATES if n_states is None else n_states
    self.absorbing_state = self.n_states
    self.get_states()
    self.get_moves()
    self.get_moves_d()
    self.reset()
    self.r_l = r_l
    self.r_r = R_RIGHT
    print(self.n_states)

  def get_moves(self):
    self.moves = [EMPTY_MOVE]

  def get_moves_d(self):
    self.moves_d = {s: self.moves for s in self.states}

  def get_states(self):
    self.states = list(range(self.n_states)) + [self.absorbing_state]

  def sample_shift(self):
    return np.sign(np.random.random() - P_LEFT)

  def step(self, action):
    if self.state == self.n_states:
      return self.state, R_STEP, True, {}
    shift = self.sample_shift()
    new_state = self.state + shift
    if not (0 <= new_state < self.n_states):
      r = self.r_r if (new_state == self.n_states) else self.r_l
      return self.n_states, r, True, {}
    self.state = new_state
    return self.state, R_STEP, False, {}

  def force_state(self, state):
    self.state = state

  def reset(self):
    self.state = self.n_states // 2
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def __str__(self):
    return self.state


class NotSoRandomWalk:
  def __init__(self, n_states=None, r_l=-1):
    self.n_states = N_STATES if n_states is None else n_states
    self.absorbing_state = self.n_states
    self.get_states()
    self.get_moves()
    self.get_moves_d()
    self.reset()
    self.r_l = r_l
    self.r_r = R_RIGHT

  def get_moves(self):
    self.moves = [LEFT, RIGHT]

  def get_moves_d(self):
    self.moves_d = {s: self.moves for s in self.states}

  def get_states(self):
    self.states = list(range(self.n_states)) + [self.absorbing_state]

  def sample_shift(self, action):
    return np.sign(np.random.random() - P_LEFT_L[action])

  def step(self, action):
    new_state = self.state + self.sample_shift(action) 
    if not (0 <= new_state < self.n_states):
      r = self.r_r if (new_state == self.n_states) else self.r_l
      return self.n_states, r, True, {}
    self.state = new_state
    return self.state, R_STEP, False, {}

  def reset(self):
    self.state = self.n_states // 2
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def __str__(self):
    return self.state
