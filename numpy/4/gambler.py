from mdp import MDP

LOSE = 0
R_LOSE = 0
R_WIN = 1


class GamblerEnv(MDP):
  def __init__(self, size=100, p_heads=0.4):
    # size of the problem is the goal in dollars
    self._goal = size
    self._p_heads = p_heads
    super().__init__()

  @property
  def p_heads(self):
    return self._p_heads

  @property
  def goal(self):
    return self._goal

  @property
  def moves(self):
    return list(range(self.goal + 1))

  @property
  def states(self):
    return list(range(LOSE, self.goal + 1))

  @property
  def r(self):
    return [R_LOSE, R_WIN]

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    capital, stakes, target_capital = s, a, s_p
    if self.is_terminal(s):
      return float((r == 0) and (s_p == s))
    if (stakes > min(capital, self.goal - capital) or
        (target_capital not in [(s + stakes), (s - stakes)])
        or (r == R_WIN and target_capital != self.goal)
        or (r == R_LOSE) and target_capital == self.goal):
        return 0
    return (self.p_heads * (target_capital >= s) +
            (1 - self.p_heads) * (target_capital <= s))

  def is_terminal(self, s):
    return s == LOSE or s == self.goal
