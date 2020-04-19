import numpy as np
from td import TD
import copy

class TDAfterstate(TD):
  def __init__(self, env, V_init=None, step_size=None, gamma=0.9, eps=0.1):
    super().__init__(env, V_init, step_size, gamma)
    self.eps = eps
    self.reset()

  def generate_traj(self, pi, log_act=False):
    s = self.env.reset()
    traj = []
    while True:
      a = self.pi[s]
      s_p, r, done, _ = self.env.step(a)
      traj.append((s, r) if not log_act else (s, a, r))
      s = s_p
      if done:
        return traj + [(s_p, 0) if not log_act else (s_p, 0, 0)]

  def update_pi(self, s):
    self.pi[s] = self.eps_greedy(s)

  def eps_greedy(self, s, eps=None):
    eps = self.eps if eps is None else eps
    moves = self.env.moves_d[s]
    if np.random.random() < self.eps:
      return moves[np.random.randint(len(moves))]
    vals = [self.V_as[self.env.after_state(s, a)] for a in moves]
    diff = np.max(vals) - np.min(vals)
    dist = (vals - np.min(vals)) / diff if diff != 0 else [1 / len(vals) for _ in range(len(vals))]
    dist /= np.sum(dist)
    return moves[np.random.choice(np.arange(len(moves)), p=dist)]

  def td0_afterstate(self, pi, n_episodes):
    self.pi = pi
    self.eps *= 0.99
    for ep_nb in range(n_episodes):
      if ep_nb > 0 and ep_nb % 100 == 0:
        print(ep_nb)
      traj = self.generate_traj(pi, log_act=True)
      for i in range(len(traj) - 1):
        (s, a, r), (s_p, a_p, _) = traj[i], traj[i + 1]
        s_as, s_p_as = self.env.after_state(s, a), self.env.after_state(s_p, a_p)
        self.V_as[s_as] += self.step_size * (self.V_as[s_p_as] - self.V_as[s_as])
        self.update_pi(s)

  def afterstate_control(self, n_episodes):
    for _ in range(n_episodes):
      self.update_pi()
      self.td0_afterstate(self.pi, 10)

  def get_V(self):
    return {s: self.V_as[self.env.after_state(s, self.eps_greedy(s, eps=0))] for s in self.env.states}

  def reset(self):
    super().reset()
    self.V_as = {s: np.random.random() for s in self.env.states}
