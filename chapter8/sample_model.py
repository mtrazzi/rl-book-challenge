class SampleModel:
  def __init__(self, env):
    self.env = env

  def sample_s_r(self, s, a):
    self.env.force_state(s)
    s_p, r, _, _ = self.env.step(a)
    return s_p, r 
