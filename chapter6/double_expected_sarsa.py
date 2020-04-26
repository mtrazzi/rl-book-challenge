import copy
import numpy as np
from expected_sarsa import ExpectedSarsa 

class DoubleExpectedSarsa(ExpectedSarsa): 
  def __init__(self, env, step_size=0.1, gamma=1, eps=0.1, pol_deriv=None):
    super().__init__(env, step_size, gamma, eps, pol_deriv) 
    self.greedy_pol = self.eps_gre(eps=0)
    self.reset()
    print(f"gamma={self.gamma}") 
    print(f"eps={eps}") 
    print(f"step_size={self.step_size}") 

  def double_expected_sarsa_update(self, s, a, r, s_p):
    (Q_1, Q_2), (pi_1, pi_2) = (self.Q_l, self.pi_l) if np.random.random() < 0.5 else (self.Q_l[::-1], self.pi_l[::-1])
    # pi_dist_1 replace the a_max_Q_1
    pi_dist_1 = np.array([self.pi[(a, s_p)] for a in self.env.moves_d[s_p]])
    # update Q_1 accordingly
    Q_1[(s, a)] += self.step_size * (r + np.dot(pi_dist_1, [Q_2[(s_p, a)] for a in self.env.moves_d[s_p]]) - Q_1[(s, a)])
    # update corresponding pi
    self.update_pi(pi_1, Q_1, s)
    # update the Q sum
    self.update_Q(s, a)
    # update the sum policy
    self.update_pi(self.pi, self.Q, s)

  def double_expected_sarsa_log_actions(self, n_episodes, to_log_s, to_log_a):
    per_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      nb_a, nb_s = 0, 0
      while True:
        a = self.pol_deriv(s)
        s_p, r, d, _ = self.env.step(a) 
        nb_s += (s == to_log_s)
        nb_a += (a == to_log_a) * (s == to_log_s)
        self.double_expected_sarsa_update(s, a, r, s_p)
        if d:
          per_l.append(100 * (nb_a / nb_s))
          break
        s = s_p
    return per_l

  def update_Q(self, s, a):
    self.Q[(s, a)] = sum(Q[(s, a)] for Q in self.Q_l)
 
  def reset(self):
    super().reset()
    self.Q_l = [copy.deepcopy(self.Q) for _ in range(2)]
    self.pi_l = [copy.deepcopy(self.pi) for _ in range(2)]
