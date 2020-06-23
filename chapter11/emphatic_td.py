import numpy as np
from tdc import TDC


class EmphaticTD(TDC):
  def __init__(self, env, pi, b, w_dim, alpha, gamma, vhat, nab_vhat,
               M_0, I_0, feat):
    super().__init__(env, pi, b, w_dim, alpha, None, gamma, vhat, feat)
    self.M = M_0
    self.int = I_0
    self.nab_vhat = nab_vhat
    self.n_st = len(self.env.states)
    self.reset()

  def exp_val_s_a(self, s, a):
    return np.sum([self.env.p[(s_p, r, s, a)] * (r + self.g *
                                                 self.vhat(s_p, self.w))
                   for s_p in self.env.states for r in self.env.r])

  def exp_val_s(self, s):
    return np.sum([self.pi[(a, s)] * self.exp_val_s_a(s, a)
                  for a in self.env.moves])

  def pol_eva(self, n_sweeps):
    S, A = self.env.states, self.env.moves
    for _ in range(n_sweeps):
      self.mu += 1
      w_delta = np.zeros_like(self.w).astype(np.float64)
      for s in S:
        td_err = np.sum([self.b[(a, s)] * self.exp_val_s_a(s, a)
                         for a in A]) - self.vhat(s, self.w)
        w_delta += self.M * td_err * self.nab_vhat(s, self.w)
      self.M = self.g * self.M + self.int
      self.w += (self.a / (len(S) * len(A))) * w_delta
    self.mu /= self.mu.sum()

  def reset(self):
    self.w = np.zeros(self.w_dim, dtype=np.float64)
    self.mu = np.zeros(len(self.env.states))
