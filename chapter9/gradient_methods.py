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

  def get_value_list(self, vhat):
    return [vhat(s, self.w) for s in self.env.states]

  def reset(self):
    self.w = np.zeros(self.d)
    self.mu = np.zeros(len(self.env.states))


class GradientMC(GradientAlg):
  def __init__(self, env, alpha, w_dim):
    super().__init__(env, alpha, w_dim)

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, gamma):
    for run in range(n_ep):
      if run > 0 and run % 1000 == 0:
        print(f"run #{run}")
      for (s, G) in gen_traj_ret(self.env, pi, gamma):
        self.mu[s] += 1
        self.w += self.a * (G - vhat(s, self.w)) * nab_vhat(s, self.w)
    self.mu /= self.mu.sum()


class SemiGradientTD0(GradientAlg):
  def __init__(self, env, alpha, w_dim):
    super().__init__(env, alpha, w_dim)

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, gamma):
    for run in range(n_ep):
      if run > 0 and run % 100 == 0:
        print(f"run #{run}")
      traj = gen_traj(self.env, pi, inc_term=True)
      for i in range(len(traj) - 1):
        (s, r), (s_p, _) = traj[i], traj[i + 1]
        self.mu[s] += 1
        self.w += self.a * (r + gamma * vhat(s_p, self.w) - vhat(s, self.w)) * nab_vhat(s, self.w)
    self.mu /= self.mu.sum()
