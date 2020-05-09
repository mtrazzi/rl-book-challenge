import copy

class Model:
  def __init__(self, states=None, moves_d=None):
    self.states = set() if states is None else set(states)
    self.moves_d = {s: set() for s in self.states} if moves_d is None else moves_d
    self.trans = {}

  def add_transition(self, s, a, r, s_p):
    self.states.add(s)
    if s in self.moves_d:
      self.moves_d[s].add(a)
    else:
      self.moves_d[s] = set([a])
    self.trans[(s, a)] = (s_p, r)

  def add_transition_cheat(self, s, a, r, s_p, moves):
    self.states.add(s)
    self.moves_d[s] = set(moves)
    for a_p in moves:
      self.trans[(s, a_p)] = (s_p, r) if a == a_p else (s, 0)

  def sample_s_r(self, s, a):
    try:
      return self.trans[(s, a)]
    except KeyError:
      print(f"transition {s}, {a} doesn't exist yet")
      import ipdb; ipdb.set_trace()

  def __str__(self):
    strg = ''
    for s in self.states:
      strg += str(s) + '\n'
      for a in self.moves_d[s]:
        s_r = f"(s_p={str(self.trans[(s, a)][0])}, r={self.trans[(s, a)][1]})" if (s, a) in self.trans else '???'
        strg += f"-{a} --> {s_r}\n"
    return strg

  def reset(self):
    self.states = set()
    self.moves_d = {}
    self.trans = {}

class FullModel(Model):
  def __init__(self, env):
    super().__init__(env.states, env.moves_d)
    self.env = copy.copy(env)

  def sample_s_r(self, s, a):
    self.env.force_state(s)
    s_p, r, _, _ = self.env.step(a)
    return s_p, r 
