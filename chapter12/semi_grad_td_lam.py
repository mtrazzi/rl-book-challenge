import numpy as np


class SemiGradTDLam:
  def __init__(self, env, alpha, w_dim, lam, vhat, nab_vhat, gamma):
    self.env = env
    self.a = alpha
    self.lam = lam
    self.d = w_dim
    self.vhat = vhat
    self.nab_vhat = nab_vhat
    self.g = gamma
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def pol_eva(self, pi, n_ep):
    for ep in range(n_ep):
      s = self.env.reset()
      z = np.zeros(self.d, dtype=np.float32)
      while True:
        a = self.sample_action(pi, s)
        s_p, r, d, _ = self.env.step(a)
        z = self.g * self.lam * z + self.nab_vhat(s, self.w)
        td_err = r + self.g * self.vhat(s_p, self.w) - self.vhat(s, self.w)
        self.w += self.a * td_err * z
        s = s_p
        if d:
          break

  def get_value_list(self):
    return [self.vhat(s, self.w) for s in self.env.states]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)
