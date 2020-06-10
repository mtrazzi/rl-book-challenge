import numpy as np

X_0_MIN, X_0_MAX = [-0.6, -0.4]
X_MIN, X_MAX = [-1.2, 0.5]
V_0_MIN = 0
V_MAX = 0.07
V_MIN = -V_MAX
V_NULL = 0
R_STEP = -1
THR_FOR, THR_REV, ZER_THR = [1, -1, 0]


class MountainCar:
  def __init__(self):
    self.reset()
    self.get_moves()

  def bound(self, y_min, y, y_max):
    return y_min if y < y_min else min(y, y_max)

  def get_moves(self):
    self.moves = [THR_FOR, THR_REV, ZER_THR]

  def step(self, a):
    self.state[1] = self.bound(V_MIN,
                               (self.state[1] +
                                0.001 * a - 0.0025 * np.cos(3 * self.state[1])),
                               V_MAX)
    self.state[0] = self.bound(X_MIN, self.state[0] + self.state[1], X_MAX)
    if self.state[0] == X_MIN:
      self.state[1] = V_NULL
    return self.state, R_STEP, self.state[0] == X_MAX, {}

  def reset(self):
    x_0 = (X_0_MAX - X_0_MIN) * np.random.random() + X_0_MIN
    self.state = [x_0, V_0_MIN]
    return self.state

  def __str__(self):
    return f"x = {self.state[0]}, vx = {self.state[1]}"
