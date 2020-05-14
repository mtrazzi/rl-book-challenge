import copy
import numpy as np
from utils import sample

START_STATE = 0
TERMINAL_STATE = -1
LEFT = 0
RIGHT = 1
P_TERM = 0.1

class Task:
  def __init__(self, b, n_states, eps=P_TERM):
    self.b = b 
    self.n_states = n_states
    self.get_states()
    self.get_moves_d()
    self.get_transitions()
    self.eps = eps

  def get_states(self):
    self.states = [TERMINAL_STATE] + list(range(self.n_states))

  def get_moves_d(self):
    self.moves_d = {s: [LEFT, RIGHT] for s in self.states}

  def sample_next_states(self):
    states_copy = copy.copy(self.states)
    next_states = []
    n_states = len(self.states)
    for i in range(self.b):
      s_idx = np.random.randint(n_states - i)
      next_states.append(states_copy[s_idx])
      del states_copy[s_idx]
    return next_states

  def get_transitions(self): 
    self.trans = {}
    for s in self.states:
      for a in self.moves_d[s]:
        exp_rew = np.random.randn()
        next_states = self.sample_next_states()
        self.trans[(s, a)] = exp_rew, next_states

  def step(self, a):
    s = self.state 
    exp_rew, next_states = self.trans[(s, a)]
    done = np.random.random() < self.eps
    s_p = TERMINAL_STATE if done else sample(next_states)
    self.state = s_p
    return s_p, exp_rew, done, {}

  def seed(self, seed):
    np.random.seed(seed)

  def reset(self):
    self.state = START_STATE
    return self.state

  def force_state(self, s):
    self.state = s
