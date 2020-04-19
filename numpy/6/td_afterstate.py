import numpy as np
from td import TD
import copy

class TDAfterstate(TD):
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9, eps=0.1, pi_init=None):
    super().__init__(env, V_init, step_size, gamma)
    self.eps = eps
    self.pi = pi_init
    self.b = {(a, s): 1 / len(env.moves_d[s]) for s in env.states for a in env.moves_d[s]}
    self.reset()

  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves_d[s]]
    return self.env.moves_d[s][np.random.choice(np.arange(len(self.env.moves_d[s])), p=pi_dist)]

  def generate_traj(self):
    s = self.env.reset()
    traj = []
    while True:
      a = self.sample_action(self.b, s) if np.random.random() < self.eps else self.pi[s]
      #a = self.sample_action(self.b, s)
      s_p, r, done, _ = self.env.step(a)
      traj.append((s, a, r))
      s = s_p
      if done:
        return traj + [(s_p, 0, 0)]

  def update_pi(self, s, a_0):
    for a in self.env.moves_d[s]:
      self.pi[(a, s)] = (a == a_0)

  def td0_afterstate(self, n_episodes):
    for ep_nb in range(n_episodes):
      traj = self.generate_traj()
      for i in range(len(traj) - 1):
        (s, a, r), (s_p, a_p, _) = traj[i], traj[i + 1]
        s_as, s_p_as = self.env.after_state(s, a), self.env.after_state(s_p, a_p)
        self.V_as[s_as] += self.step_size * (r + self.gamma * self.V_as[s_p_as] - self.V_as[s_as])

  def td0_afterstate_batch(self, n_episodes):
    self.experience = []
    for ep_nb in range(n_episodes):
      td_error_sum = {s: 0 for s in self.V_as}
      self.experience.append(self.generate_traj())
      for traj in self.experience:
        for i in range(len(traj) - 1):
          (s, a, r), (s_p, a_p, _) = traj[i], traj[i + 1]
          s_as, s_p_as = self.env.after_state(s, a), self.env.after_state(s_p, a_p)
          is_ratio = (1 - self.eps) * ((a == self.pi[s]) / self.b[(a, s)]) + self.eps * (1 - (a == self.pi[s])) / self.b[(a, s)] 
          td_error_sum[s_as] += self.step_size * is_ratio * (r + self.gamma * self.V_as[s_p_as] - self.V_as[s_as])
          #td_error_sum[s_as] += self.step_size * (r + self.gamma * self.V_as[s_p_as] - self.V_as[s_as])
      for s in self.V_as:
        self.V_as[s] += td_error_sum[s]
  
  def policy_improvement(self):
    policy_stable = True
    for s in self.env.states:
      a_old = self.pi[s]
      V_vect = np.array([self.V_as[self.env.after_state(s, a)] for a in self.env.moves_d[s]])
      a_new = self.env.moves_d[s][np.random.choice(np.flatnonzero(V_vect == V_vect.max()))]
      self.pi[s] =  a_new
      policy_stable = policy_stable and (a_old == a_new)
    return policy_stable

  def policy_iteration(self, ep_per_eval=10, batch=True, max_ep=np.inf):
    pi_log = [str(self.pi)]
    pol_eval = self.td0_afterstate_batch if batch else self.td0_afterstate
    while True: 
      print(len(pi_log))
      pol_eval(ep_per_eval)
      pol_stable = self.policy_improvement()
      pi_str = str(self.pi)
      if pol_stable or pi_str in pi_log or len(pi_log) >= max_ep:
        if pol_stable:
          print("stable")
        return self.V_as, self.pi, pol_stable
      pi_log.append(pi_str)

  def reset(self):
    super().reset()
    self.V_as = {s: np.random.random() for s in self.env.states}
