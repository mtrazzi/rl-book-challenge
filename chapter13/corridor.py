S_0, S_1, S_2, S_G, = 0, 1, 2, 3
L, R = -1, 1
R_STEP = -1

KEY_ACTION_DICT = {
  's': L,
  'f': R,
}

class Corridor:
  def __init__(self):
    self.get_states()
    self.get_moves()
    self.get_keys()
    self.reset()

  def get_states(self):
    self.states = [S_0, S_1, S_2, S_G]

  def get_moves(self):
    self.moves = [L, R]

  def get_keys(self):
    self.keys = KEY_ACTION_DICT.keys()

  def step(self, a):
    if self.state == S_1:
      a = L if a == R else R
    self.state = max(0, self.state + a)
    return self.state, R_STEP, self.state == S_G, {}

  def step_via_key(self, key):
    return self.step(KEY_ACTION_DICT[key])

  def seed(self, seed):
    pass

  def reset(self):
    self.state = S_0
    return self.state

  def __str__(self):
    return f"S={self.state}"
