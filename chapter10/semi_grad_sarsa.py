import numpy as np
from utils import sample


class GradientAlg:
  def __init__(self, env, alpha, w_dim, eps):
    self.env = env
    self.a = alpha
    self.d = w_dim
    self.eps = eps
    self.qhat = None
    self.reset()

  def gre(self, s):
    q_arr = np.array([self.qhat(s, a, self.w) for a in self.env.moves])
    best_move = np.random.choice(np.flatnonzero(q_arr == q_arr.max()))
    return self.env.moves[best_move]

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.env.moves_d[s])
    return self.gre(s)

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)


class EpisodicSemiGradientTD0(GradientAlg):
  def __init__(self, env, alpha, w_dim, eps):
    super().__init__(env, alpha, w_dim, eps)

  def pol_eva(self, qhat, nab_qhat, n_ep, gamma):
    steps_per_ep = []
    self.qhat, w = qhat, self.w
    for ep in range(n_ep):
      if ep > 0 and ep % 1 == 0:
        print(f"ep #{ep}")
      s = self.env.reset()
      a = self.eps_gre(s)
      n_steps = 0
      while True:
        s_p, r, d, _ = self.env.step(a)
        n_steps += 1
        if d:
          self.w += self.a * (r - qhat(s, a, w)) * nab_qhat(s, a, w)
          break
        a_p = self.eps_gre(s_p)
        self.w += self.a * ((r + gamma * qhat(s_p, a_p, w) - qhat(s, a, w)) *
                            nab_qhat(s, a, w))
        s, a = s_p, a_p
      steps_per_ep.append(n_steps)
    return steps_per_ep
