from mdp import MDP
import random

R_WIN = 1
R_STEP = 0
R_RIGHT = 0
LEFT = 0
RIGHT = 1
S_INIT = 0
S_ABS = 1
P_STAY = 0.9

class OneState(MDP):
  def __init__(self):
    super().__init__()
    self.reset()

  def seed(self, seed=0):
    random.seed(seed)

  @property
  def moves(self):
    return [LEFT, RIGHT]

  @property
  def states(self):
    return [S_INIT, S_ABS]

  @property
  def r(self):
    return [R_STEP, R_WIN]

  def step(self, action):
    if self.state == S_ABS:
      return S_ABS, R_STEP, True, {}
    if action == LEFT:
      if random.random() < P_STAY:
        return S_INIT, R_STEP, False, {}
      else:
        return S_ABS, R_WIN, True, {}
    else:
      return S_ABS, R_RIGHT, True, {}

  def force_state(self, s):
    self.state = s
    return

  def reset(self):
    self.state = S_INIT
    return S_INIT

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    pass

  def __str__(self):
    return f"{self.state}"
