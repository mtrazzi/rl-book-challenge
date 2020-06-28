import numpy as np
from utils import gen_traj


class Reinforce:
  def __init__(self, env, alpha, gamma, the_d, pi_gen, log_pi, the_0=None):
    self.env = env
    self.a = alpha
    self.g = gamma
    self.the_d = the_d
    self.pi_gen = pi_gen
    self.log_pi = log_pi
    self.the_0 = the_0
    self.reset()

  def train(self, n_ep):
    env, alp, the, g = self.env, self.a, self.the, self.g
    logpi, pi_gen = self.log_pi, self.pi_gen
    tot_rew_l = []
    for _ in range(n_ep):
      pi = pi_gen(env, the)
      S, A, R = zip(*gen_traj(env, pi))
      T = len(S)
      g_l = [g ** k for k in range(T)]
      for t in range(T):
        G = np.dot(g_l[:T-t], R[:T-t])
        the += alp * g_l[t] * G * logpi(A[t], S[t], pi)
      tot_rew_l.append(np.sum(R))
    return tot_rew_l

  def reset(self):
    self.the = np.zeros(self.the_d) if self.the_0 is None else self.the_0

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)
