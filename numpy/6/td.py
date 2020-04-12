import numpy as np
import copy

class TD:
  def __init__(self, env, gamma=0.9):
    self.env = env
    self.gamma = gamma

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)


class OneStepTD(TD):
  def __init__(self, env, V_init=None, step_size=0.1, gamma=0.9):
    super().__init__(env, gamma)
    self.step_size = step_size
    self.V_init = V_init
    self.reset()
  
  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)), p=pi_dist)]
 
  def generate_traj(self, pi):
    s = self.env.reset()
    traj = []
    while True:
      s_p, r, done, _ = self.env.step(self.sample_action(pi, s))
      traj.append((s, r))
      s = s_p
      if done:
        return traj + [(s_p, 0)]
 
  def tabular_td_0(self, pi):
    traj = self.generate_traj(pi)
    for i in range(len(traj) - 1):
      (s, r), (s_p, _) = traj[i], traj[i + 1]
      self.V[s] += self.step_size * (r + self.gamma * self.V[s_p] - self.V[s])

  def constant_step_size_mc(self, pi):
    traj = self.generate_traj(pi)
    G = 0
    for (s, r) in traj[::-1]:
      G += r
      self.V[s] += self.step_size * (G - self.V[s])

  def get_value_list(self):
    return [val for key,val in self.V.items()]


  def reset(self): 
    self.V = {s: 0 for s in self.env.moves} if self.V_init is None else copy.deepcopy(self.V_init)
