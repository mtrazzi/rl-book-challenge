import numpy as np
import time

class TrajectorySampling:
  def __init__(self, env, gamma=1, eps=0.1):
    self.env = env
    self.g = gamma
    self.eps = eps
    self.reset()

  def gre(self, s):
    q_arr = np.array([self.Q[(s, a)] for a in self.env.moves_d[s]])
    return self.env.moves_d[s][np.random.choice(np.flatnonzero(q_arr == q_arr.max()))]

  #def estimate_state(self, s_0, pi=None):
  #  if pi is None:
  #    a_max = self.gre(s_0)
  #    moves = self.env.moves_d[s_0]
  #    p_a = self.eps / len(moves)
  #    pi = {(s_0, a): p_a + (1 - self.eps) * (a == a_max) for a in moves}
  #  return sum(pi[(s_0, a)] * self.Q[(s_0, a)] for a in moves)

  def gre_estimation(self, s):
    r_sum, d = 0, False
    self.env.force_state(s)
    while not d:
      s, r, d, _ = self.env.step(self.gre(s))
      r_sum += r
    return r_sum

  def eps_gre(self, s):
    if np.random.random() < self.eps:
      return sample(self.env.moves_d[s])
    return self.gre(s)

  def exp_update(self, s, a):
    exp_rew, next_states = self.env.trans[(s, a)]
    self.Q[(s, a)] = exp_rew + self.g * (1 - self.eps) * sum(max(self.Q[(s_p, a_p)] for a_p in self.env.moves_d[s_p]) for s_p in next_states)

  def uniform(self, start_state, n_updates, log_freq=100):
    values = []
    start = time.time()
    for upd in range(n_updates):
      for s in self.env.states:
        for a in self.env.moves_d[s]:
          self.exp_update(s, a) 
      if upd % log_freq == (log_freq - 1):
        print(f"{upd + 1} updates (total of {time.time()-start:.2f}s)")
        values.append(self.gre_estimation(start_state))
    return values

  def reset(self):
    self.Q = {(s, a): 0 for s in self.env.states for a in self.env.moves_d[s]} 
