import copy

class Model:
  def __init__(self, states, moves_d):
    self.states = states
    self.moves_d = moves_d
    self.trans = {}

  def add_transition(self, s, a, r, s_p):
    self.trans[(s, a)] = (s_p, r)

  def sample_s_r(self, s, a):
    try:
      return self.trans[(s, a)]
    except KeyError:
      print(f"transition {s}, {a} doesn't exist yet")

  def reset(self):
    self.trans = {}

class SampleModel(Model):
  def __init__(self, env):
    self.env = copy.copy(env)
    self.states = env.states
    self.moves_d = env.moves_d

  def sample_s_r(self, s, a):
    self.env.force_state(s)
    s_p, r, _, _ = self.env.step(a)
    return s_p, r 
