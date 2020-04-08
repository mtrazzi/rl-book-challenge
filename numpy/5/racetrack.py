from mdp import MDP
import numpy as np
import pandas as pd
import random

from ipdb import set_trace as d

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

  def __eq__(self, other_vel):
    return self.x == other_vel.x and self.y == other_vel.y

  def __str__(self):
    return f"({self.x}, {self.y})"

class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y
 
  def __eq__(self, other_pos):
    return self.x == other_pos.x and self.y == other_pos.y

  def __str__(self):
    return f"({self.x}, {self.y})"

class RaceState:
  def __init__(self, pos, vel):
    self.p = pos
    self.v = vel

  def __str__(self): 
    return f"pos={self.p}, vel={self.v}"

  def __eq__(self, other_state):
    return self.p == other_state.p# and self.v == other_state.v

  def __hash__(self):
    return hash((self.p.x, self.p.y, self.v.x, self.v.y))

  def is_valid(self, race_map):
    return (self.p in race_map.valid_pos
            and (self.v.x > 0 or self.v.y > 0
                 or self in race_map.initial_states))

class RaceMap:
  def __init__(self, filename):
    """reads file and builds corresponding map"""
    self.file_arr = np.array(pd.read_csv(filename))
    self.build_map() 

  def build_map(self):
    self.grid = self.grid_from_lines()
    self.extreme_left = self.get_extreme_left()
    self.fill_grid()
    print(self.grid)
    print(self.extreme_left)
    print([str(state) for state in self.initial_states])
    print([str(pos) for pos in self.valid_pos])

  @property
  def valid_pos(self):
    valid_pos_list = []
    for x in range(self.grid.shape[0]):
      for y in range(self.grid.shape[1]):
        valid_pos_list.append(Position(x, y))
    return valid_pos_list

  @property
  def initial_states(self):
    y_0 = abs(self.extreme_left)
    return [RaceState(Position(y_0, y_0 + i), Velocity(0, 0)) for i in range(self.file_arr[0, 2])]

  def get_extreme_left(self):
    """Returns position of extreme left from first rectangle."""
    pos = min_y = 0
    for (shift, _, _) in self.file_arr:
      pos += shift
      if pos < min_y:
        min_y = pos
    return min_y

  def fill_grid(self):
    x, y = 0, abs(self.extreme_left)
    for (shift, n_rows, n_cols) in self.file_arr:
      y += shift
      self.grid[x:x + n_rows, y:y + n_cols] = True
      x += n_rows

  def grid_from_lines(self):
    y_shifts = self.file_arr[:, 0]
    y_left, y_right = [np.dot(y_shifts, (y_shifts * sign) > 0) for sign in [-1, 1]]
    max_width = self.file_arr[:, 2].max()
    height = self.file_arr[:, 1].sum()
    grid_map = np.zeros((y_right - y_left + max_width, height))  
    return grid_map

class RacetrackEnv(MDP):
  def __init__(self, filename):
    self.race_map = RaceMap(filename)
    self.reset()

  def seed(self, seed=0):
    random.seed(seed)

  @property
  def velocities(self):
    return [Velocity(*vel) for vel in VEL_LIST]

  @property
  def moves(self):
    return [(x, y) for x in VEL_CHANGES for y in VEL_CHANGES]

  @property
  def states(self):
    return [RaceState(pos, vel) for pos in self.race_map.valid_pos for vel in self.velocities if RaceState(pos, vel).is_valid(self.race_map)]

  @property
  def r(self):
    return [R_STEP]

  def update_velocities(self, action):
    pass

  def update_position(self, action):
    pass

  def step(self, action):
    return self.state, R_STEP, True, {}

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
