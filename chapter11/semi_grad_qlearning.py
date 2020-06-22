import numpy as np


class SemiGradQLearning:
  def __init__(self, env, pi, b, w_dim, alpha, gamma, qhat, nab_qhat):
    self.env = env
    self.a = alpha
    self.g = gamma
    self.pi = pi
    self.b = b
    self.w_dim = w_dim
    self.qhat = qhat
    self.nab_qhat = nab_qhat
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def qlearn_error(self, s, a, r, s_p):
    q_max = max([self.qhat(s_p, a, self.w) for a in self.env.moves])
    return r + self.g * q_max - self.qhat(s, a, self.w)

  def qlearn_update(self, s, a, r, s_p):
    is_ratio = self.pi[(a, s)] / self.b[(a, s)] if self.pi[(a, s)] > 0 else 0
    ql_err = self.qlearn_error(s, a, r, s_p)
    self.w = self.w + self.a * ql_err * is_ratio * self.nab_qhat(s, a, self.w)

  def pol_eva(self, n_steps):
    s = self.env.reset()
    for _ in range(n_steps):
      a = self.sample_action(self.b, s)
      s_p, r, d, _ = self.env.step(a)
      self.qlearn_update(s, a, r, s_p)
      s = s_p if not d else self.env.reset()

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.w_dim)
