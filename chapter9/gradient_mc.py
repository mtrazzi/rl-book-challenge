import numpy as np

GAM_UND = 1

class GradientMC:
  def __init__(self, env, alpha, w_dim, gamma=GAM_UND):
    self.env = env
    self.a = alpha
    self.d = w_dim
    self.g = gamma

  def sample_action(self, pi, s):
    moves = self.env.moves_d[s]
    pi_dist = [pi[(a, s)] for a in moves]
    return self.env.moves_d[s][np.random.choice(np.arange(len(moves)), p=pi_dist)]

  def gen_traj(self, pi): 
    traj, d, s = [], False, self.env.reset()
    while not d:
      s_p, r, d, _ = self.env.step(self.sample_action(pi, s))
      traj.append((s, r))
      s = s_p
    return traj

  def gen_traj_ret(self, pi):
    traj = self.gen_traj(pi)
    ret_traj, G = [], 0
    for (t, (s, r)) in enumerate(traj[::-1]):
      G = r + self.g * G
      ret_traj.append((s, G))
    return ret_traj[::-1]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def pol_eva(self, pi, vhat, nab_vhat, n_ep, w_0=None):
    self.w = np.zeros(self.d) if w_0 is None else w_0
    for _ in range(n_ep):
      for (S_t, G_t) in self.gen_traj(pi):
        self.w += self.a * (G_t - vhat(S_t, self.w)) * nab_vhat(S_t, self.w)

