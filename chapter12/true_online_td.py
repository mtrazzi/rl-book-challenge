import numpy as np


class TrueOnlineTD:
  def __init__(self, env, alpha, w_dim, lam, vhat, nab_vhat, gamma):
    self.env = env
    self.a = alpha
    self.lam = lam
    self.d = w_dim
    self.vhat = vhat
    self.feat = lambda s, w: nab_vhat(s, w) * (s != self.env.absorbing_state)
    self.g = gamma
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def pol_eva(self, pi, n_ep):
    w, alp, g, lam, env = self.w, self.a, self.g, self.lam, self.env
    sample, feat = self.sample_action, self.feat
    for _ in range(n_ep):
      s = env.reset()
      z = np.zeros(self.d, dtype=np.float32)
      v_old = 0
      while True:
        a = sample(pi, s)
        s_p, r, d, _ = self.env.step(a)
        x, x_p = [feat(st, w) for st in [s, s_p]]
        v, v_p = [np.dot(w, u) for u in [x, x_p]]
        td_err = r + g * v_p - v
        z = g * lam * z + (1 - alp * g * lam * np.dot(z, x)) * x
        w += alp * (td_err + v - v_old) * z - alp * (v - v_old) * x
        v_old = v_p
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
