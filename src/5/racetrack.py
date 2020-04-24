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
NOISE_PROB = 0.1

class Velocity:
  def __init__(self, v_x, v_y):
    self.x = v_x 
    self.y = v_y

  def norm(self):
    return np.linalg.norm([self.x, self.y])

  def __eq__(self, other_vel):
    return self.x == other_vel.x and self.y == other_vel.y

  def __add__(self, other_vel):
    return Velocity(self.x + other_vel.x, self.y + other_vel.y)

  def __hash__(self):
    return hash((self.x, self.y))

  def __str__(self):
    return f"({self.x}, {self.y})"

class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def updated_pos(self, vel):
    return Position(self.x + vel.x, self.y + vel.y)
  
  def is_valid_pos(self, grid):
    return ((0 <= self.x < grid.shape[0]) and
            (0 <= self.y < grid.shape[1]) and
            grid[self.x, self.y])
 
  def __eq__(self, other_pos):
    return self.x == other_pos.x and self.y == other_pos.y

  def __add__(self, other_pos):
    return Position(self.x + other_pos.x, self.y + other_pos.y)

  def __str__(self):
    return f"({self.x}, {self.y})"

class RaceState:
  def __init__(self, pos, vel):
    self.p = pos
    self.v = vel

  def __str__(self): 
    return f"pos={self.p}, vel={self.v}"

  def __eq__(self, other_state):
    return self.p == other_state.p and self.v == other_state.v
  
  def __hash__(self):
    return hash((self.p.x, self.p.y, self.v.x, self.v.y))

  def is_valid(self, race_map):
    return (self.p.is_valid_pos(race_map.grid) and (self.v.x > 0 or self.v.y > 0
             or self in race_map.initial_states))

class RaceMap:
  def __init__(self, filename):
    """Reads file and builds corresponding map."""
    self.file_arr = np.array(pd.read_csv(filename))
    self.build_map() 

  def build_map(self):
    self.y_min, self.y_max = self.get_extremes() 
    self.grid = np.zeros((self.file_arr[:,1].sum(), self.y_max - self.y_min))
    self.fill_grid()
    print(self.grid)
    self.get_initial_states() 
    self.get_valid_pos_and_finish()

  def get_valid_pos_and_finish(self):
    self.valid_pos, self.finish_line = [], []
    for x in range(self.grid.shape[0]):
      for y in range(self.grid.shape[1]):
        if self.grid[x, y]:
          self.valid_pos.append(Position(x, y)) 
          if y == self.grid.shape[1] - 1:
            self.finish_line.append(Position(x, y))

  def get_initial_states(self):
    y_0 = abs(self.y_min)
    self.initial_states = [RaceState(Position(0, y_0 + i),Velocity(0, 0)) for i in range(self.file_arr[0, 2])]

  def get_extremes(self):
    """Returns position of extreme left from first rectangle."""
    pos = min_y = 0
    max_y = pos + self.file_arr[0, 2]
    for (shift, _, n_cols) in self.file_arr:
      pos += shift
      if pos < min_y:
        min_y = pos
      if pos + n_cols > max_y:
        max_y = pos + n_cols
    return min_y, max_y

  def fill_grid(self):
    x, y = 0, abs(self.y_min)
    for (shift, n_rows, n_cols) in self.file_arr:
      y += shift
      self.grid[x:x + n_rows, y:y + n_cols] = True
      x += n_rows

class RacetrackEnv:
  def __init__(self, filename, noise=True):
    self.race_map = RaceMap(filename)
    self.get_velocities()
    self.get_states()
    self.get_moves()
    self.noise = noise
    self.reset()

  def seed(self, seed=0):
    random.seed(seed)
    np.random.seed(seed)

  def get_velocities(self):
    self.velocities = [Velocity(*vel) for vel in VEL_LIST]

  def get_moves(self):
    self.moves = [Velocity(x, y) for x in VEL_CHANGES for y in VEL_CHANGES]

  def get_states(self):
    self.states = [RaceState(pos, vel) for pos in self.race_map.valid_pos for vel in self.velocities if RaceState(pos, vel).is_valid(self.race_map)]

  @property
  def r(self):
    return [R_STEP]

  def is_valid_vel(self, vel):
    return VEL_MIN <= vel.x <= VEL_MAX and VEL_MIN <= vel.y <= VEL_MAX

  def intersections(self, s_s, s_e):
    # returns all the states intersected between start and end state
    x, y = s_s.p.x, s_s.p.y
    delta_x, delta_y = s_e.p.x - x, s_e.p.y - y
    dx, dy = np.sign(delta_x), np.sign(delta_y) 
    if (abs(delta_x) == abs(delta_y)) or dx == 0 or dy == 0:
      return [Position(x + i * dx, y + i * dy) for i in range(max(abs(delta_x), abs(delta_y)) + 1)]
    else:
      return [Position(x + i * dx, y + j * dy) for i in range(abs(delta_x) + 1) for j in range(abs(delta_y) + 1)]

  def will_hit_boundary(self, s_s, s_e):
    # returns True if car will hit boundaries between start and end state
    return not np.all([p.is_valid_pos(self.race_map.grid) for p in self.intersections(s_s, s_e)])

  def step(self, action):
    # noise to make task more challenging (disabling at test time)
    if self.noise and np.random.random() < NOISE_PROB:
      action = Velocity(VEL_MIN, VEL_MIN)
    # tolerance for wrong vel actions
    new_vel = self.state.v + action
    if not self.is_valid_vel(new_vel):
      return self.state, R_STEP, False, {}
    # if wrong end goal or will hit boundary, go to start line
    new_pos = self.state.p.updated_pos(new_vel)
    new_state = RaceState(new_pos, new_vel)
    if (not new_state.is_valid(self.race_map)) or self.will_hit_boundary(self.state, new_state):
      return self.reset(), R_STEP, False, {}
    self.state = new_state
    return new_state, R_STEP, new_pos in self.race_map.finish_line, {}

  def force_state(self, s):
    self.state = s
    return

  def sample_init_state(self):
    init_states = self.race_map.initial_states
    rand_idx = np.random.randint(len(init_states))
    return init_states[rand_idx]

  def reset(self):
    self.state = self.sample_init_state()
    return self.state

  def __str__(self):
    return f"{self.state}"
