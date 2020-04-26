import numpy as np

INIT_POS = (3, 0)
GOAL_POS = (3, 11)
GRID_SHAPE = (4, 12)
R_STEP = -1
R_CLIFF = -100
KEY_ACTION_DICT = {
  'z': (-1, 0),
  'q': (0, -1),
  'd': (0, 1),
  's': (1, 0),
}
POS_CHAR_DICT = {
  GOAL_POS: 'G',
  INIT_POS: 'S',
}
AGENT_KEY = 'A'
CLIFF_KEY = 'C'

class Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  def in_bounds(self, index, axis):
    return max(0, min(index, GRID_SHAPE[axis] - 1))

  def in_cliff(self):
    return self.x == (GRID_SHAPE[0] - 1) and 0 < self.y < GRID_SHAPE[1] - 1
  
  def next_state(self, action): 
    s_p = Position(self.in_bounds(self.x + action[0], 0), self.in_bounds(self.y + action[1], 1))
    return (Position(*INIT_POS), R_CLIFF) if s_p.in_cliff() else (s_p, R_STEP)

  def __eq__(self, other_pos):
    if isinstance(other_pos, tuple):
      return self.x == other_pos[0] and self.y == other_pos[1]
    return self.x == other_pos.x and self.y == other_pos.y

  def __hash__(self):
    return hash((self.x, self.y))

  def __str__(self):
    return f"({self.x}, {self.y})"

class TheCliff:
  def __init__(self):
    self.get_states()
    self.get_moves()
    self.get_moves_dict()
    self.get_keys()

  def get_moves(self):
    self.moves = [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (abs(x) + abs(y)) == 1]

  def get_states(self):
    self.states = [Position(x, y) for x in range(GRID_SHAPE[0]) for y in range(GRID_SHAPE[1])]

  def get_moves_dict(self):
    self.moves_d = {s: [a for a in self.moves if s.next_state(a)[0] in self.states] for s in self.states}

  def step(self, action):
    self.state, r = self.state.next_state(action)
    return self.state, r, self.state == Position(*GOAL_POS), {}

  def get_keys(self):
    self.keys = KEY_ACTION_DICT.keys()

  def step_via_key(self, key):
    return self.step(KEY_ACTION_DICT[key])

  def reset(self):
    self.state = Position(*INIT_POS)
    return self.state

  def seed(self, seed):
    pass

  def __str__(self):
    x_ag, y_ag = self.state.x, self.state.y
    s = ''
    s += '\n'
    for x in range(GRID_SHAPE[0]):
      for y in range(GRID_SHAPE[1]):
        if (x, y) == (x_ag, y_ag):
          s += AGENT_KEY
        elif (x, y) in POS_CHAR_DICT.keys():
          s += POS_CHAR_DICT[(x, y)]
        elif Position(x, y).in_cliff():
          s += CLIFF_KEY
        else:
          s += '.'
      s += '\n'
    return s
