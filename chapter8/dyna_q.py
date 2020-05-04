import numpy as np
from tabular_q import TabularQ
from utils import sample
from models import Model

class DynaQ(TabularQ):
  def __init__(self, env, alpha, gamma, eps):
    super().__init__(Model(env.states, env.moves_d), alpha, gamma)
    self.env = env
    self.eps = eps
    self.S = env.states
    self.A = env.moves_d
    self.reset()

  def best_actions(self, s):
    q_max, a_max_l = -np.inf, []
    for a in self.A[s]:
      q_val = self.Q[(s, a)]
      if q_val >= q_max:
        a_max_l = [a] if q_val > q_max else a_max_l + [a]
        q_max = q_val
    return a_max_l

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.A[s])
    return sample(self.best_actions(s))

  def q_learning_update(self, s, a, r, s_p):
    Q_max = self.Q[(s_p, self.best_actions(s).pop())]
    self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])

  def tabular_dyna_q(self, n_eps, n_upd=1):
    ep_len_l = []
    for _ in range(n_eps):
      s = self.env.reset() 
      n_steps = 0
      while True:
        n_steps += 1
        a = self.eps_gre(s)
        s_p, r, d, _ = self.env.step(a)
        self.q_learning_update(s, a, r, s_p)
        self.model.add_transition(s, a, r, s_p)
        self.rand_sam_one_step_pla(n_upd)
        if d:
          ep_len_l.apend(n_steps)
          break
    return ep_len_l

  def reset(self): 
    super().reset()
    self.model.reset()
