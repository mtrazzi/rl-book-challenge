import numpy as np

R_FROM_A = 0
R_ABS = 0
LEFT = 0
RIGHT = 1
ABS_ACT = 0
S_A = 0
S_B = 1
S_INIT = S_A
S_ABS = 2
N_ACT_FROM_B = 100

class MaxBiasMDP:
  def __init__(self):
    self.reset()
    self.moves_d = {S_A: [LEFT, RIGHT], S_B: [i for i in range(N_ACT_FROM_B)], S_ABS: [ABS_ACT]}
    self.states = [S_A, S_B, S_ABS]

  def seed(self, seed=0):
    random.seed(seed)

  def step(self, a):
    if self.s == S_A:
      r = R_FROM_A
      s_p, d = (S_ABS, True) if a == RIGHT else (S_B, False)
    elif self.s == S_B:
      s_p, r, d = S_ABS, np.random.randn() - 1, True
    else:
      s_p, r, d = S_ABS, R_ABS, True
    self.s = s_p
    return s_p, r, d, {}

  def reset(self):
    self.s = S_INIT
    return S_INIT

  def seed(self, seed):
    np.random.seed(seed) 

  def __str__(self):
    return f"{self.state}"
