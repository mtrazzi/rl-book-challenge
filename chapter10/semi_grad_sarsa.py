import numpy as np

class GradientAlg:
  def __init__(self, env, alpha, w_dim, eps):
    self.env = env
    self.a = alpha
    self.d = w_dim
    self.eps = eps
    self.reset()

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.env.moves_d[s])
    return self.gre(s)

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)
    self.mu = np.zeros(len(self.env.states))

class EpisodicSemiGradientTD0(GradientAlg):
  def __init__(self, env, alpha, w_dim, eps):
    super().__init__(env, alpha, w_dim, eps)

  def pol_eva(self, qhat, nab_qhat, n_ep, gamma):
    steps_per_ep = []
    for ep in range(n_ep):
      if ep > 0 and ep % 100 == 0:
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
        self.w += self.a * (r + gamma * qhat(s_p, a_p, w) - qhat(s, a, w)) * nab_qhat(s, a, w)
        s, a = s_p, a_p
      steps_per_ep.append(n_steps) 
    return steps_per_ep
