from abc import ABC, abstractmethod


class MDP(ABC):
    """Base environment for the mdp envirnoment of the MC chapter."""

    def __init__(self):
        self.init_p()

    def init_p(self):
        self.p = {(s_p, r, s, a): self._p(s_p, r, s, a)
                  for s in self.states for a in self.moves
                  for s_p in self.states for r in self.r}

    @abstractmethod
    def _p(self, s_p, r, s, a):
        """Specific transition probabilities for environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def states(self):
        """List of possible states."""
        raise NotImplementedError

    @property
    @abstractmethod
    def r(self):
        """List of possible rewards."""
        raise NotImplementedError

    @property
    @abstractmethod
    def moves(self):
        """List of all available actions."""
        raise NotImplementedError
