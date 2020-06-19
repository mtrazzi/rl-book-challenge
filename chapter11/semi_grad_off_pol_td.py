import numpy as np


class SemiGradOffPolTD:
  def __init__(self, env, pi, b, w_dim, alpha, gamma, vhat, nab_vhat):
    self.env = env
    self.a = alpha
    self.g = gamma
    self.pi = pi
    self.b = b
    self.w_dim = w_dim
    self.vhat = vhat
    self.nab_vhat = nab_vhat
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def td_error(self, s, r, s_p):
    return r + self.g * self.vhat(s_p, self.w) - self.vhat(s, self.w)

  def generate_episode(self):
    return self.generate_traj(self.b, log_act=True)

  def off_policy_td_update(self, s, a, r, s_p):
    is_ratio = self.pi[(a, s)] / self.b[(a, s)] if self.pi[(a, s)] > 0 else 0
    td_err = self.td_error(s, r, s_p)
    self.w = self.w + self.a * td_err * is_ratio * self.nab_vhat(s, self.w)

  def pol_eva(self, n_steps):
    s = self.env.reset()
    for _ in range(n_steps):
      a = self.sample_action(self.b, s)
      s_p, r, d, _ = self.env.step(a)
      self.off_policy_td_update(s, a, r, s_p)
      s = s_p if not d else self.env.reset()

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.w_dim)
