import numpy as np

class TD:
  def __init__(self, env, gamma=0.9):
    self.env = env
    self.gamma = gamma

  def seed(self, seed):
    self.env.seed(seed)
    np.random.seed(seed)


class OneStepTD(TD):
  def __init__(self, env, step_size=0.1, gamma=0.9):
    super().__init__(env, gamma)
    self.step_size = step_size
    self.reset()
  
  def sample_action(self, pi, s):
    pi_dist = [pi[(a, s)] for a in self.env.moves]
    return self.env.moves[np.random.choice(np.arange(len(self.env.moves)), p=pi_dist)]
  
  def tabular_td_0(self, pi): 
    s = self.env.reset()
    while True:
      a = self.sample_action(pi, s)
      s_p, r, done, _ = self.env.step(a)
      print(f"{self.V[s]} += {self.step_size} * ({r} + {self.gamma} * {self.V[s_p]} - {self.V[s]})")
      self.V[s] += self.step_size * (r + self.gamma * self.V[s_p] - self.V[s])
      s = s_p
      if done:
        break
  
  def reset(self): 
    self.V = {s: 0 for s in self.env.moves}
