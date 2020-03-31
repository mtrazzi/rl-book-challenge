from mdp import MDP
import random

R_LOSE = -1
R_DRAW = 0
R_WIN = 1
R_STEP = 0
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
N_CARD_DEALER = 1
N_CARD_PLAYER = 2


class Player:
  def __init__(self, n_initial_cards):
    """Player starts with two, dealer with one hidden / one visible."""
    self.n_initial_cards = n_initial_cards
    self.cards = []
    self.sum = 0
    self.reset()

  def sample_card(self):
    return random.randint(ACE_HIGH, NB_VALUES)

  def cards_to_values(self, card):
    """Replace the value of kings, queens, jacks to 10."""
    return min(card, 10)

  def new_card(self):
    self.cards.append(self.sample_card())
    new_card_value = self.cards_to_values(self.cards[-1])
    # TODO: maybe some edge cases below with two aces etc.
    if self.sum + new_card_value <= BLACKJACK:
      self.sum += self.cards_to_values(self.cards[-1])
    else:
      self.bust = True

  def update_aces(self):
    # TODO: maybe some edge cases below with two aces etc.
    self.usable_ace = (ACE_LOW in self.cards)
    if self.usable_ace and (self.player_sum + ACE_DIFF <= BLACKJACK):
      self.sum += ACE_DIFF

  def reset(self):
    for _ in range(self.n_initial_cards):
      self.new_card()
    self.update_aces()


class BlackjackEnv(MDP):
  def __init__(self):
    super().__init__()
    self.players = {"dealer": Player(N_CARD_DEALER),
                    "player": Player(N_CARD_PLAYER)}

  @property
  def moves(self):
    return [STICK, HIT]

  @property
  def states(self):
    # states are encoded using: player's sum, dealer's showing card * ace or not
    return list(range((BLACKJACK - MIN_SUM + 1) * MAX_DEAL_CARD * ACE_STATES))

  @property
  def r(self):
    return [R_LOSE, R_DRAW, R_WIN]

  def is_natural(self):
    return self.player_won() and self.usable_ace

  def get_result(self):
    sum_diff = self.players["player"].sum - self.players["dealer"].sum
    if sum_diff > 0:
      return R_WIN
    elif sum_diff == 0:
      return R_DRAW
    else:
      return R_LOSE

  def hit(self):
    s = self.get_state()
    self.players["player"].new_card()
    done = not self.players["player"].bust
    if done:
      return s, R_LOSE, done, {}
    return self.get_state(), R_STEP, done, {}

  def stick(self):
    self.players['dealer'].new_card()
    return self.get_state(), self.get_result(), True, {}

  def step(self, action):
    if self.is_natural():
      return self.get_state(), R_WIN, True, {}
    return self.do_hit() if action == HIT else self.do_stick()

  def compute_state(self, player_sum, usable_ace, dealer_card):
    nb_player_sums = BLACKJACK - MIN_SUM + 1
    nb_dealer_scores = MAX_DEAL_CARD
    return (usable_ace * nb_player_sums * nb_dealer_scores +
            nb_player_sums * (dealer_card - 1) +
            (player_sum - 1))

  def get_state(self):
    return self.compute_state(self.players['player'].sum,
                              self.players['player'].usable_ace,
                              self.players['dealer'].cards[0])

  def reset(self):
    for player in self.players.values():
      player.reset()

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    pass
