import numpy as np


class TDC:
  def __init__(self, env, pi, b, w_dim, alpha, beta, gamma, vhat, feat):
    self.env = env
    self.a = alpha
    self.g = gamma
    self.pi = pi
    self.b = b
    self.bet = beta
    self.w_dim = w_dim
    self.vhat = vhat
    self.feat = feat
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)),
                                           p=pi_dist)]

  def td_error(self, s, r, s_p):
    return r + self.g * self.vhat(s_p, self.w) - self.vhat(s, self.w)

  def lms_rule(self, is_ratio, td_err, xt, vx):
    return self.v + self.bet * is_ratio * (td_err - vx) * xt

  def w_update(self, is_ratio, td_err, xt, vx, xtp1):
    return self.w + self.a * is_ratio * (td_err * xt - self.g * xtp1 * vx)

  def tdc_update(self, s, a, r, s_p):
    is_r = self.pi[(a, s)] / self.b[(a, s)] if self.pi[(a, s)] > 0 else 0
    td_err = self.td_error(s, r, s_p)
    xt, xtp1 = self.feat(s), self.feat(s_p)
    vx = np.dot(self.v, xt)
    comm_args = (is_r, td_err, xt, vx)
    self.v, self.w = self.lms_rule(*comm_args), self.w_update(*comm_args, xtp1)

  def pol_eva(self, n_steps):
    s = self.env.reset()
    for _ in range(n_steps):
      self.mu[s - 1] += 1
      a = self.sample_action(self.b, s)
      s_p, r, d, _ = self.env.step(a)
      self.tdc_update(s, a, r, s_p)
      s = s_p if not d else self.env.reset()

  def ve(self, vpi):
    sq_err = [(vpi(s) - self.vhat(s, self.w)) ** 2 for s in self.env.states]
    return np.dot(self.mu, sq_err)

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.w_dim)
    self.v = np.zeros(self.w_dim)
    self.mu = np.zeros(len(self.env.states))
