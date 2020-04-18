import numpy as np
from sarsa import Sarsa

class ExpectedSarsa(Sarsa):
  def __init__(self, env, step_size=0.1, gamma=1, eps=0.1, update_pi=None):
    super().__init__(env, step_size, gamma, eps, None) 
    self.update_pi = self.update_policy(eps) if update_pi is None else update_pi
    self.reset()
    print(f"alpha={self.step_size}")
    print(f"gamm={gamma}")
    print(f"eps={eps}")

  def uniform_pol(self, env):
    return {(a, s): 1 / len(env.moves_d[s]) for s in env.states for a in env.moves_d[s]}

  def pi_dist(self, s):
    return np.array([self.pi[(a, s)] for a in self.env.moves_d[s]])

  def sample_action_d(self, s):
    return self.env.moves_d[s][np.random.choice(np.arange(len(self.env.moves_d[s])), p=self.pi_dist(s))]

  def expected_sarsa_update(self, s, a, r, s_p):
    self.Q[(s, a)] += self.step_size * (r + np.dot(self.pi_dist(s_p), [self.Q[s_p, a] for a in self.env.moves_d[s_p]]) - self.Q[(s, a)])

  def update_policy(self, eps):
    def update_on_q_values(pi, Q, s):
      best_a = self.eps_gre(0)(s)
      moves = self.env.moves_d[s]
      soft_min = eps / len(moves)
      for a in moves:
        pi[(a, s)] = soft_min + (1 - eps) * (a == best_a)
    return update_on_q_values

  def expected_sarsa(self, n_episodes): 
    r_sum_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      r_sum = 0
      while True:
        a = self.sample_action_d(s)
        s_p, r, d, _ = self.env.step(a) 
        r_sum += r
        a_p = self.sample_action_d(s)
        self.expected_sarsa_update(s, a, r, s_p)
        self.update_pi(self.pi, self.Q, s)
        if d:
          r_sum_l.append(r_sum)
          break
        s = s_p
    return r_sum_l

  def expected_sarsa_log_actions(self, n_episodes, to_log_s, to_log_a): 
    per_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      nb_a, nb_s = 0, 0
      while True:
        a = self.sample_action_d(s)
        s_p, r, d, _ = self.env.step(a) 
        nb_s += (s == to_log_s)
        nb_a += (a == to_log_a) * (s == to_log_s)
        a_p = self.sample_action_d(s)
        self.expected_sarsa_update(s, a, r, s_p)
        self.update_pi(self.pi, self.Q, s)
        if d:
          per_l.append(100 * (nb_a / nb_s))
          break
        s = s_p
    return per_l

  def reset(self):
    super().reset()
    self.pi = self.uniform_pol(self.env)
