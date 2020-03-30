from mdp import MDP
import random

LOSE_REW = -1
DRAW_REW = 0
WIN_REW = 1
HIT = 1
STICK = 0
MIN_SUM = 12
BLACKJACK = 21
MAX_DEAL_CARD = 10
ACE_STATES = 2
NUMBER_CARDS = 52
NUMBER_COLORS = 4
NB_VALUES = NUMBER_CARDS // NUMBER_COLORS
ACE_LOW = 1
ACE_HIGH = 11
ACE_DIFF = ACE_HIGH - ACE_LOW


class BlackjackEnv(MDP):
  def __init__(self):
    super().__init__()
    self.reset()

  @property
  def moves(self):
    return [STICK, HIT]

  @property
  def states(self):
    # states are encoded using: player's sum, dealer's showing card * ace or not
    return list(range((BLACKJACK - MIN_SUM + 1) * MAX_DEAL_CARD * ACE_STATES))

  @property
  def r(self):
    return [LOSE_REW, DRAW_REW, WIN_REW]

  def step(self, action):
    current_sum = s
    return

  def sample_card(self):
    return random.randint(ACE_HIGH, NB_VALUES)

  def cards_to_values(self, card):
    """Replace the value of kings, queens, jacks to 10."""
    return min(card, 10)

  def deal(self):
    player_cards, self.dealer_card = ([self.sample_card(), self.sample_card()],
                                      self.sample_card())
    self.player_score = map(self.cards_to_values, player_cards)
    self.dealer_score = self.cards_to_values(self.dealer_card)
    self.player_sum = sum(self.player_score)
    self.usable_ace = (ACE_LOW in player_cards)
    if self.player_sum + ACE_DIFF <= BLACKJACK:
      self.player_sum += ACE_DIFF

  def compute_state(self, player_sum, usable_ace, dealer_score):
    nb_player_sums = BLACKJACK - MIN_SUM + 1
    nb_dealer_scores = MAX_DEAL_CARD
    return (usable_ace * nb_player_sums * nb_dealer_scores +
            nb_player_sums * (dealer_score - 1) +
            (player_sum - 1))

  def reset(self):
    self.deal()
    return self.compute_state(self.player_sum, self.usable_ace,
                              self.dealer_score)

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    pass
