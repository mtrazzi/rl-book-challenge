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
    return self.lms_rule(*comm_args), self.w_update(*comm_args, xtp1)

  def pol_eva(self, n_steps):
    s = self.env.reset()
    for _ in range(n_steps):
      self.mu[self.env.states.index(s)] += 1
      a = self.sample_action(self.b, s)
      s_p, r, d, _ = self.env.step(a)
      self.v, self.w = self.tdc_update(s, a, r, s_p)
      s = s_p if not d else self.env.reset()
    self.mu /= self.mu.sum()

  def munorm(self, arr):
    return np.dot(self.mu, np.square(arr))

  def ve(self, vpi):
    return self.munorm([vpi(s) - self.vhat(s, self.w) for s in self.env.states])

  def proj_mat(self):
    S, d = len(self.env.states), self.w_dim
    X = np.zeros((S, d))
    for (idx, s) in enumerate(self.env.states):
      X[idx, :] = self.feat(s)
    D = np.diag(self.mu)
    return X @ np.linalg.pinv(X.T @ D @ X) @ X.T @ D

  def exp_val_s_a(self, s, a):
    return np.sum([self.env.p[(s_p, r, s, a)] * (r + self.g *
                                                 self.vhat(s_p, self.w))
                   for s_p in self.env.states for r in self.env.r])

  def bell_op(self, s):
    return np.sum([self.pi[(a, s)] * self.exp_val_s_a(s, a)
                  for a in self.env.moves])

  def delta_vec(self):
    return [self.bell_op(s) - self.vhat(s, self.w) for s in self.env.states]

  def pbe(self):
    return self.munorm(self.proj_mat() @ self.delta_vec())

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.w_dim)
    self.v = np.zeros(self.w_dim)
    self.mu = np.zeros(len(self.env.states))


class ExpectedTDC(TDC):
  def __init__(self, env, pi, b, w_dim, alpha, beta, gamma, vhat, feat):
    super().__init__(env, pi, b, w_dim, alpha, beta, gamma, vhat, feat)

  def pol_eva(self, n_sweeps):
    S, R, A = self.env.states, self.env.r, self.env.moves
    for sweep in range(n_sweeps):
      if sweep > 0 and sweep % 10 == 0:
        print(sweep)
      self.mu += 1
      pair_arr = [self.tdc_update(s, a, r, s_p)
                  for s in S for r in R for a in A for s_p in S]
      self.w = np.mean([w for (w, _) in pair_arr], axis=0)
      self.v = np.mean([v for (_, v) in pair_arr], axis=0)
    self.mu /= self.mu.sum()
