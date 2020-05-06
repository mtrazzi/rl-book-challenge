import numpy as np

INIT_POS = (2, 0)
GOAL_POS = (0, 8)
GRID_SHAPE = (6, 9)
KEY_ACTION_DICT = {
  'z': (-1, 0),
  'q': (0, -1),
  'd': (0, 1),
  's': (1, 0),
}
AGENT_KEY = 'A'
GOAL_KEY = 'G'
INIT_KEY = 'S'
WALL_KEY = 'W'
WALLS = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]

class Position:
  def __init__(self, x, y, walls):
    self.x = x
    self.y = y
    self.is_wall = self.get_wall(walls)
    self.is_goal = self.get_goal()

  def get_goal(self):
    return self.x == GOAL_POS[0] and self.y == GOAL_POS[1] 

  def in_bounds(self, index, axis):
    return max(0, min(index, GRID_SHAPE[axis] - 1))

  def get_wall(self, walls):
    return (self.x, self.y) in walls
  
  def next_state(self, action, walls):
    s_p = Position(self.in_bounds(self.x + action[0], 0), self.in_bounds(self.y + action[1], 1), walls)
    return self if (s_p.is_wall or self.is_goal) else s_p

  def __eq__(self, other_pos):
    if isinstance(other_pos, tuple):
      return self.x == other_pos[0] and self.y == other_pos[1]
    return self.x == other_pos.x and self.y == other_pos.y

  def __hash__(self):
    return hash((self.x, self.y))

  def __str__(self):
    return f"({self.x}, {self.y})"

class DynaMaze:
  def __init__(self, init_pos=INIT_POS, goal_pos=GOAL_POS, grid_shape=GRID_SHAPE, walls=WALLS):
    self.init_pos = init_pos
    self.goal_pos = goal_pos
    self.grid_shape = grid_shape
    self.walls = walls
    self.get_states()
    self.get_moves()
    self.get_moves_dict()
    self.get_keys()
    self.pos_char_dict = {self.init_pos: INIT_KEY, self.goal_pos: GOAL_KEY}

  def get_moves(self):
    self.moves = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (abs(x) + abs(y)) == 1]

  def get_states(self):
    self.states = [Position(x, y, self.walls) for x in range(GRID_SHAPE[0]) for y in range(GRID_SHAPE[1]) if not Position(x, y, self.walls).is_wall]

  def get_moves_dict(self):
    self.moves_d = {s: self.moves for s in self.states}

  def step(self, action):
    next_state = self.state.next_state(action, self.walls)
    done = next_state.is_goal
    r = float(done and not self.state.is_goal)
    self.state = next_state
    return next_state, r, done, {}

  def get_keys(self):
    self.keys = KEY_ACTION_DICT.keys()

  def step_via_key(self, key):
    return self.step(KEY_ACTION_DICT[key])

  def reset(self):
    self.state = Position(*self.init_pos, self.walls)
    return self.state

  def seed(self, seed):
    pass

  def force_state(self, s):
    self.state = s

  def __str__(self):
    x_ag, y_ag = self.state.x, self.state.y
    s = ''
    s += '\n'
    for x in range(GRID_SHAPE[0]):
      for y in range(GRID_SHAPE[1]):
        if (x, y) == (x_ag, y_ag):
          s += AGENT_KEY
        elif (x, y) in self.pos_char_dict.keys():
          s += self.pos_char_dict[(x, y)]
        elif Position(x, y, self.walls).is_wall:
          s += WALL_KEY
        else:
          s += '.'
      s += '\n'
    return s
