S_0, S_1, S_2, S_G = 0, 1, 2, 3
L, R = -1, 1
R_STEP = -1


class Corridor:
  def __init__(self):
    self.get_states()
    self.get_moves()
    self.reset()

  def get_states(self):
    self.states = [S_0, S_1, S_2, S_G]

  def get_moves(self):
    self.moves = [L, R]

  def step(self, a):
    d = (self.state == S_2) and (a == R)
    if self.state == S_1:
      a = L if a == R else R
    self.state = max(0, self.state + a)
    return self.state, R_STEP, d, {}

  def seed(self, seed):
    pass

  def reset(self):
    self.state = S_0
    return self.state
