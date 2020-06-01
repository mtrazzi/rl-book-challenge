import numpy as np
from utils import gen_traj, gen_traj_ret


class GradientAlg:
  def __init__(self, env, alpha, w_dim):
    self.env = env
    self.a = alpha
    self.d = w_dim
    self.reset()

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)


  def reset(self):
    self.w = np.zeros(self.d)
    self.mu = np.zeros(len(self.env.states))


class GradientMC(GradientAlg):
  def __init__(self, env, alpha, w_dim):
    super().__init__(env, alpha, w_dim)

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, gamma):
    for ep in range(n_ep):
      if ep > 0 and ep % 1000 == 0:
        print(f"ep #{ep}")
      for (s, G) in gen_traj_ret(self.env, pi, gamma):
        self.mu[s] += 1
        #print(f"w={self.w}")
        #print(G, vhat(s, self.w), nab_vhat(s, self.w))
        #input()
        self.w += self.a * (G - vhat(s, self.w)) * nab_vhat(s, self.w)
    self.mu /= self.mu.sum()

class SemiGradientTD0(GradientAlg):
  def __init__(self, env, alpha, w_dim):
    super().__init__(env, alpha, w_dim)

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, gamma):
    for ep in range(n_ep):
      if ep > 0 and ep % 100 == 0:
        print(f"ep #{ep}")
      traj = gen_traj(self.env, pi, inc_term=True)
      for i in range(len(traj) - 1):
        (s, r), (s_p, _) = traj[i], traj[i + 1]
        self.mu[s] += 1
        self.w += self.a * (r + gamma * vhat(s_p, self.w) - vhat(s, self.w)) * nab_vhat(s, self.w)
    self.mu /= self.mu.sum()
