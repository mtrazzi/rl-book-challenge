import copy
import numpy as np
from sarsa import Sarsa 

class DoubleQLearning(Sarsa): 
  def __init__(self, env, step_size=0.1, gamma=1, eps=0.1, pol_deriv=None):
    super().__init__(env, step_size, gamma, eps, pol_deriv) 
    self.greedy_pol = self.eps_gre(eps=0)
    self.reset()
    print(f"gamma={self.gamma}") 
    print(f"eps={eps}") 
    print(f"step_size={self.step_size}") 

  def double_q_learning_update(self, s, a, r, s_p):
    Q_1, Q_2 = self.Q_l if np.random.random() < 0.5 else self.Q_l[::-1]
    a_max_Q_1 = self.best_action(self.env.moves_d[s_p], np.array([Q_1[(s_p, a)] for a in self.env.moves_d[s_p]]))
    Q_1[(s, a)] += self.step_size * (r + self.gamma * Q_2[(s_p, a_max_Q_1)] - Q_1[(s, a)])

  def double_q_learning_log_actions(self, n_episodes, to_log_s, to_log_a):
    per_l = []
    for ep_nb in range(n_episodes):
      s = self.env.reset()
      nb_a, nb_s = 0, 0
      while True:
        a = self.pol_deriv(s)
        s_p, r, d, _ = self.env.step(a) 
        nb_s += (s == to_log_s)
        nb_a += (a == to_log_a) * (s == to_log_s)
        self.double_q_learning_update(s, a, r, s_p)
        self.update_Q(s, a)
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
