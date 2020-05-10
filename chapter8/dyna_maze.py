import numpy as np

INIT_POS = (2, 0)
GOAL_POS_L = [(0, 8)]
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
  def __init__(self, x, y, is_goal):
    self.x = x
    self.y = y
    self.is_goal = is_goal

  def in_bounds(self, grid_shape, index, axis):
    return max(0, min(index, grid_shape[axis] - 1))
 
  def move(self, action, grid_shape, goal_pos_l):
    x_p, y_p = self.in_bounds(grid_shape, self.x + action[0], 0), self.in_bounds(grid_shape, self.y + action[1], 1)
    return self if self.is_goal else Position(x_p, y_p, (x_p, y_p) in goal_pos_l)

  def __eq__(self, other_pos):
    if isinstance(other_pos, tuple):
      return self.x == other_pos[0] and self.y == other_pos[1]
    return self.x == other_pos.x and self.y == other_pos.y

  def __hash__(self):
    return hash((self.x, self.y))

  def __str__(self):
    return f"({self.x}, {self.y})"

class DynaMaze:
  def __init__(self, init_pos=INIT_POS, goal_pos_l=GOAL_POS_L, grid_shape=GRID_SHAPE, walls1=WALLS, walls2=WALLS):
    self.init_pos = init_pos
    self.goal_pos_l = goal_pos_l
    self.grid_shape = grid_shape
    self.get_states()
    self.get_moves()
    self.get_moves_dict()
    self.get_keys()
    self.pos_char_dict = {pos: INIT_KEY if pos == self.init_pos else GOAL_KEY for pos in [self.init_pos] + self.goal_pos_l}
    self.walls = walls1
    self.walls1 = walls1
    self.walls2 = walls2

  def switch_walls(self):
    if self.walls == self.walls1:
      self.walls = self.walls2
    else:
      self.walls = self.walls1

  def get_moves(self):
    self.moves = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (abs(x) + abs(y)) == 1]

  def get_states(self):
    self.states = [Position(x, y, (x, y) in self.goal_pos_l) for x in range(self.grid_shape[0]) for y in range(self.grid_shape[1])]

  def get_moves_dict(self):
    self.moves_d = {s: self.moves for s in self.states}

  def step(self, action):
    s_curr = self.state
    s_p = s_curr.move(action, self.grid_shape, self.goal_pos_l)
    s_next = s_curr if (s_p.x, s_p.y) in self.walls else s_p
    done = s_next.is_goal
    r = float(done and not s_curr.is_goal)
    self.state = s_next
    return s_next, r, done, {}

  def get_keys(self):
    self.keys = KEY_ACTION_DICT.keys()

  def step_via_key(self, key):
    return self.step(KEY_ACTION_DICT[key])

  def reset(self):
    self.state = Position(*self.init_pos, self.init_pos in self.goal_pos_l)
    return self.state

  def seed(self, seed):
    pass

  def force_state(self, s):
    self.state = s

  def __str__(self):
    x_ag, y_ag = self.state.x, self.state.y
    s = ''
    s += '\n'
    for x in range(self.grid_shape[0]):
      for y in range(self.grid_shape[1]):
        if (x, y) == (x_ag, y_ag):
          s += AGENT_KEY
        elif (x, y) in self.pos_char_dict.keys():
          s += self.pos_char_dict[(x, y)]
        elif (x, y) in self.walls:
          s += WALL_KEY
        else:
          s += '.'
      s += '\n'
    return s
