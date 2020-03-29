from mdp import MDP


class BlackjackEnv(MDP):
    def __init__(self):
    super().__init__()

  @property
  def moves(self):
    return []

  @property
  def states(self):
    return []

  @property
  def r(self):
    return []

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
