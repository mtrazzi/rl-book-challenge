from dyna_q import DynaQ, show
from utils import sample
import numpy as np
import time

class DynaQPlus(DynaQ):
  def __init__(self, env, alpha, gamma, eps, k):
    super().__init__(env, alpha, gamma, eps)
    self.k = k
    self.reset()

  def plan_dyna_q_plus(self, n_updates=1):
    s_list = list(self.model.states)
    a_dict = {s: list(self.env.moves_d[s]) for s in s_list}
    for _ in range(n_updates):
      s = sample(s_list)
      a = sample(a_dict[s]) 
      s_p, r_raw = self.model.sample_s_r(s, a) if a in self.model.moves_d[s] else (s, 0)
      r = r_raw + self.k * np.sqrt(self.trans_count[(s, a)])
      Q_max = max(self.Q[(s_p, a_p)] for a_p in a_dict[s])
      self.Q[(s, a)] += self.a * (r + self.g * Q_max - self.Q[(s, a)])

  def upd_count(self, s_0, a_0):
    for s in self.env.states:
      for a in self.env.moves_d[s]:
        self.trans_count[(s, a)] += 1
    self.trans_count[(s_0, a_0)] = 0

  def tabular_dyna_q_step(self, n_steps=1, n_plan_steps=1):
    cum_rew_l = []
    cum_rew = 0
    s = self.env.reset()
    for step in range(n_steps):
      a = self.eps_gre(s)
      self.upd_count(s, a)
      s_p, r, d, _ = self.env.step(a)
      self.q_learning_update(s, a, r, s_p)
      self.model.add_transition(s, a, r, s_p)
      self.plan_dyna_q_plus(n_plan_steps)
      s = self.env.reset() if d else s_p
      cum_rew += r
      cum_rew_l.append(cum_rew)
    return cum_rew_l

  def print_trans_count(self):
    for s in self.env.states:
      print(s)
      for a in self.env.moves_d[s]:
        print(f"->{a}: count={self.trans_count[(s, a)]}, Q={self.Q[(s, a)]}")

  def reset(self):
    super().reset()
    self.trans_count = {(s, a): 0 for s in self.env.states for a in self.env.moves_d[s]}
