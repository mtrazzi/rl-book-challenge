import numpy as np
from utils import sample


class TrueOnlineSarsa:
  def __init__(self, env, alpha, w_dim, lam, F, qhat, eps, gamma):
    self.env = env
    self.a = alpha
    self.lam = lam
    self.d = w_dim
    self.qhat = qhat
    self.g = gamma
    self.eps = eps
    self.feat = self.feat_from_F(F)
    self.g = gamma
    self.reset()

  def feat_from_F(self, F):

    def feat(s, a):
      z = np.zeros(self.d)
      for idx in F(s, a):
        z[idx] = 1
      return z
    return feat

  def gre(self, s):
    q_arr = np.array([self.qhat(s, a, self.w) for a in self.env.moves])
    best_move = np.random.choice(np.flatnonzero(q_arr == q_arr.max()))
    return self.env.moves[best_move]

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.env.moves)
    return self.gre(s)

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def pol_eva(self, pi, n_ep, max_steps=np.inf):
    def act(s): return (self.eps_gre(s) if pi is None else
                        self.sample_action(pi, s))
    w, alp, g, lam, env, wd = self.w, self.a, self.g, self.lam, self.env, self.d
    feat, env = self.feat, self.env
    step_l = []
    for _ in range(n_ep):
      s = env.reset()
      a = act(s)
      z = np.zeros(wd, dtype=np.float32)
      q_old = n_steps = 0
      while True and n_steps < max_steps:
        s_p, r, d, _ = env.step(a)
        n_steps += 1
        a_p = act(s_p)
        x, x_p = [feat(st, ac) for (st, ac) in [(s, a), (s_p, a_p)]]
        if d:
          x_p = np.zeros(wd)
        q, q_p = [np.dot(w, u) for u in [x, x_p]]
        td_err = r + g * q_p - q
        z = g * lam * z + (1 - alp * g * lam * np.dot(z, x)) * x
        w += alp * (td_err + q - q_old) * z - alp * (q - q_old) * x
        q_old = q_p
        s, a = s_p, a_p
        if d:
          break
      step_l.append(n_steps)
    return step_l

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)
