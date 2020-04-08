from mdp import MDP
import numpy as np
import pandas as pd
import random

VEL_CHANGES = [-1, 0, 1]
VEL_MIN = 0
VEL_MAX = 5
VEL_RANGE = range(VEL_MIN, VEL_MAX + 1) 
VEL_LIST = [(x, y) for x in VEL_RANGE for y in VEL_RANGE]
R_STEP = -1

class Velocity:
  def __init__(self, v_x, v_y):
    self.x = v_x 
    self.y = v_y

class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y

class RaceState:
  def __init__(self, pos, vel):
    self.p = pos
    self.v = vel

  def is_valid(self, racemap):
    return self.p in racemap.valid_pos_list

class RaceMap:
  def __init__(self, filename):
    """reads file and builds corresponding map"""
    self.build_map(np.array(pd.read_csv(filename)))

  def build_map(self, pos_list):
    self.valid_pos_list = pos_list
    self.initial_states = pos_list[0]

  def rectangles_from_lines(self, file_arr):
    import ipdb; ipdb.set_trace()
    y_shifts = file_arr[:, 0]
    y_left, y_right = [np.dot(y_shifts, (y_shifts * sign) > 0) for sign in [-1, 1]]
    height = file_arr.shape[0]
    grid_map = np.zeros((x_right - x_left + 1, height)) 

class RacetrackEnv(MDP):
  def __init__(self, filename):
    self.race_map = RaceMap(filename)
    super().__init__()
    self.reset()

  def seed(self, seed=0):
    random.seed(seed)

  @property
  def moves(self):
    return [(x, y) for x in VEL_CHANGES for y in VEL_CHANGES]

  @property
  def states(self):
    return [RaceState(pos, vel) for pos in self.race_map.valid_pos_list for vel in VEL_LIST if RaceState(pos, vel).is_valid(self.race_map)]

  @property
  def r(self):
    return [R_STEP]

  def update_velocities(self, action):
    pass

  def update_position(self, action):
    pass

  def step(self, action):
    return 0, 0, 0, {}

  def force_state(self, s):
    self.state = s
    return

  def reset(self):
    init_states = self.race_map.initial_states
    rand_idx = np.random.randint(len(init_states))
    self.state = self.race_map.initial_states[rand_idx]
    return self.state

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    pass

  def __str__(self):
    return f"{self.state}"
