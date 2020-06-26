import numpy as np
from utils import sample


class SarsaLam:
  def __init__(self, env, alpha, w_dim, lam, F, qhat, eps, gamma):
    self.env = env
    self.a = alpha
    self.lam = lam
    self.d = w_dim
    self.F = F
    self.qhat = qhat
    self.g = gamma
    self.eps = eps
    self.reset()

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

  def pol_eva(self, pi, n_ep, acc=True):
    def act(s): return (self.eps_gre(s) if pi is None else
                        self.sample_action(pi, s))
    w, dim, alp, g, lam = self.w, self.d, self.a, self.g, self.lam
    F, env = self.F, self.env
    step_list = []
    for _ in range(n_ep):
      s = self.env.reset()
      a = act(s)
      z = np.zeros(dim, dtype=np.float32)
      n_steps = 0
      while True:
        print(np.mean(self.w))
        s_p, r, d, _ = env.step(a)
        n_steps += 1
        delt = r
        for i in F(s, a):
          delt += w[i]
          z[i] = (z[i] + 1) if acc else 1
        if d:
          w += alp * delt * z
          break
        a_p = act(s_p)
        for i in F(s_p, a_p):
          w += alp * delt * z
          z = g * lam * z
          s, a = s_p, a_p
      step_list.append(n_steps)
    return step_list

  def get_value_list(self):
    return [self.vhat(s, self.w) for s in self.env.states]

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)

  def reset(self):
    self.w = np.zeros(self.d)
