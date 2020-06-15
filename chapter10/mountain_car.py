import numpy as np
import matplotlib.pyplot as plt

X_0_MIN, X_0_MAX = [-0.6, -0.4]
X_MIN, X_MAX = [-1.2, 0.5]
V_0 = 0
V_MAX = 0.07
V_MIN = -V_MAX
V_NULL = 0
R_STEP = -1
THR_FOR, THR_REV, ZER_THR = [1, -1, 0]

KEY_ACTION_DICT = {
  's': THR_REV,
  'd': ZER_THR,
  'f': THR_FOR,
}


class MountainCar:
  def __init__(self):
    self.get_moves()
    self.get_keys()

  def bound(self, y_min, y, y_max):
    return y_min if y < y_min else min(y, y_max)

  def get_moves(self):
    self.moves = [THR_REV, ZER_THR, THR_FOR]

  def step(self, a):
    self.state[1] = self.bound(V_MIN,
                               (self.state[1] +
                                0.001 * a - 0.0025 * np.cos(3 * self.state[0])),
                               V_MAX)
    self.state[0] = self.bound(X_MIN, self.state[0] + self.state[1], X_MAX)
    if self.state[0] == X_MIN:
      self.state[1] = V_NULL
    return self.state, R_STEP, self.state[0] == X_MAX, {}

  def get_keys(self):
    self.keys = KEY_ACTION_DICT.keys()

  def step_via_key(self, key):
    return self.step(KEY_ACTION_DICT[key])

  def reset(self):
    x_0 = (X_0_MAX - X_0_MIN) * np.random.random() + X_0_MIN
    self.state = [x_0, V_0]
    return self.state

  def seed(self, seed):
    np.random.seed(seed)

  def show(self, n_pts=50):
    X = np.linspace(X_MIN, X_MAX, n_pts)
    Y = [-np.cos(3 * x) for x in X][::-1]
    V = -np.array(Y[::-1])
    plt.plot(X, V, 'k', label='v')

    x0, dx = n_pts // 5, n_pts // 5 + n_pts // 10
    v0 = np.argmin(np.abs(V[x0:x0 + dx])) + x0
    y0 = np.argmin(Y)
    phi = X[v0] - X[y0]
    Y_phi = [-np.cos(3 * (x + phi)) for x in X][::-1]
    plt.plot(X, Y_phi)

    x_idx = np.flatnonzero(X >= self.state[0])[0]
    plt.plot(X[x_idx], Y_phi[x_idx], 'r+', label='car')
    plt.plot(X[x_idx], V[x_idx], 'rx', label='vx')
    plt.plot(X, np.zeros(n_pts), '--k', label='v=0')
    plt.legend()
    plt.draw()
    plt.pause(1)
    plt.close()

  def __str__(self):
    return f"x = {self.state[0]} {'<' if (self.state[0] < 0.70) else '>'} X_CRITIC, vx = {self.state[1]}"
