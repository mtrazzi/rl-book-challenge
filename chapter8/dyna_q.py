import numpy as np
from tabular_q import TabularQ
from utils import sample
from models import Model
import time

class DynaQ(TabularQ):
  def __init__(self, env, alpha, gamma, eps):
    super().__init__(Model(), alpha, gamma)
    self.env = env
    self.eps = eps
    self.reset()

  def best_actions(self, s):
    q_max, a_max_l = -np.inf, []
    for a in self.env.moves_d[s]:
      q_val = self.Q[(s, a)]
      if q_val >= q_max:
        a_max_l = [a] if q_val > q_max else a_max_l + [a]
        q_max = q_val
    return a_max_l

  def gre(self, s):
    q_arr = np.array([self.Q[(s, a)] for a in self.env.moves_d[s]])
    return self.env.moves_d[s][np.random.choice(np.flatnonzero(q_arr == q_arr.max()))]
    

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.env.moves_d[s])
    return self.gre(s)

  def q_learning_update(self, s, a, r, s_p):
    max_val = -np.inf
    Q_max = max(self.Q[(s_p, a_p)] for a_p in self.env.moves_d[s])
    self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])

  def Q_reset(self):
    for s in self.env.states:
      for a in self.env.moves_d[s]:
        self.Q[(s, a)] = 0

  def tabular_dyna_q(self, n_eps, n_plan_steps=1):
    ep_len_l = []
    for ep in range(n_eps):
      s = self.env.reset() 
      n_steps = 0
      while True:
        n_steps += 1
        a = self.eps_gre(s)
        s_p, r, d, _ = self.env.step(a)
        self.q_learning_update(s, a, r, s_p)
        self.model.add_transition(s, a, r, s_p)
        self.rand_sam_one_step_pla(n_plan_steps)
        s = s_p
        if d:
          ep_len_l.append(n_steps)
          break
    return ep_len_l

  def tabular_dyna_q_step(self, n_steps=1, n_plan_steps=1):
    cum_rew_l = []
    cum_rew = 0
    s = self.env.reset()
    for step in range(n_steps):
      a = self.eps_gre(s)
      s_p, r, d, _ = self.env.step(a)
      self.q_learning_update(s, a, r, s_p)
      self.model.add_transition_cheat(s, a, r, s_p, self.env.moves_d[s])
      self.rand_sam_one_step_pla(n_plan_steps)
      s = self.env.reset() if d else s_p
      cum_rew += r
      cum_rew_l.append(cum_rew)
    return cum_rew_l

  def test_n_steps(self, max_steps, tol=0.1): 
    s = self.env.reset()
    n_steps = 0
    d = False
    while n_steps < ((1 + tol) * max_steps) and not d:
      s, _, d, _ = self.env.step(self.gre(s))
      n_steps += 1
    return d

  def updates_until_optimal(self, n_steps_opt, n_plan_steps, tol=0.1):
    n_updates = 0
    s = self.env.reset()
    while True:
      a = self.eps_gre(s)
      s_p, r, d, _ = self.env.step(a)
      self.q_learning_update(s, a, r, s_p)
      self.model.add_transition(s, a, r, s_p)
      self.rand_sam_one_step_pla(n_plan_steps)
      s = self.env.reset() if d else s_p
      n_updates += n_plan_steps + 1
      if d and self.test_n_steps(n_steps_opt, tol):
        return n_updates

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self): 
    self.model.reset()
    self.Q_reset()
