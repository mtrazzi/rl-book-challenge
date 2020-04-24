from mdp import MDP
import random

R_LOSE = -1
R_DRAW = 0
R_WIN = 1
R_STEP = 0
STICK = 0
HIT = 1
MIN_PLAY_SUM = 12
MIN_DEAL_CARD = 1
BLACKJACK = 21
N_DEAL_SCORES = 10
ACE_STATES = 2
NUMBER_CARDS = 52
NUMBER_COLORS = 4
NB_VALUES = NUMBER_CARDS // NUMBER_COLORS
ACE_LOW = 1
ACE_HIGH = 11
ACE_DIFF = ACE_HIGH - ACE_LOW
DEAL_THRES = 17
PLAY_THRES = 12
N_POSSIBLE_PLAY_SUMS = BLACKJACK - MIN_PLAY_SUM + 1


class Player:
  def __init__(self, stick_threshold):
    """
    Deal cards so player sum is already greater than 12 (because it doesn't
    make sense to stick when sum is lower) and dealer played his
    policy of sticking when sum is lower than 17.
    """
    self.stick_threshold = stick_threshold

  @property
  def sum(self):
    return self.card_sum + ACE_DIFF * self.usable_ace

  @property
  def usable_ace(self):
    return (self.card_sum + ACE_DIFF <= BLACKJACK) and (ACE_LOW in self.cards)

  @property
  def card_sum(self):
    return sum(self.card_value(i) for i in range(len(self.cards)))

  @property
  def bust(self):
    return self.sum > BLACKJACK

  def sample_card(self):
    return random.randint(ACE_LOW, NB_VALUES)

  def cards_to_values(self, card):
    """Replace the value of kings, queens, jacks to 10."""
    return min(card, 10)

  def card_value(self, idx):
    """Returns the value of the card #idx."""
    return self.cards_to_values(self.cards[idx])

  def new_card(self):
    self.cards.append(self.sample_card())

  def deal_cards(self):
    while self.sum < self.stick_threshold:
      self.new_card()

  def reset(self, initial_cards=None):
    self.cards = [] if initial_cards is None else initial_cards
    self.deal_cards()

  def __str__(self):
    return (f"CARDS={self.cards}, " +
            f"SUM={self.sum}, " +
            f"USABLE_ACE={self.usable_ace}")


class BlackjackEnv(MDP):
  def __init__(self):
    super().__init__()
    self.players = {"dealer": Player(DEAL_THRES),
                    "player": Player(PLAY_THRES)}

  def seed(self, seed=0):
    random.seed(seed)

  @property
  def moves(self):
    return [STICK, HIT]

  @property
  def states(self):
    # states are encoded using: player's sum, dealer's showing card * ace or not
    return list(range((BLACKJACK - MIN_PLAY_SUM + 1) *
                      N_DEAL_SCORES * ACE_STATES))

  @property
  def r(self):
    return [R_LOSE, R_DRAW, R_WIN]

  def is_natural(self):
    return (self.players['player'].sum == BLACKJACK
            and len(self.players['player'].cards) == 2)

  def get_result(self):
    player, dealer = self.players["player"], self.players["dealer"]
    sum_diff = player.sum - dealer.sum
    if player.bust:
      return R_LOSE
    elif dealer.bust or sum_diff > 0:
      return R_WIN
    elif sum_diff == 0:
      return R_DRAW
    else:
      return R_LOSE

  def hit(self):
    s = self.get_state()
    self.players['player'].new_card()
    done = self.players['player'].bust
    if done:
      return s, R_LOSE, done, {}  # returning s is arbitrary
    return self.get_state(), R_STEP, done, {}

  def stick(self):
    return self.get_state(), self.get_result(), True, {}

  def step(self, action):
    if self.is_natural():
      return self.get_state(), R_WIN, True, {}
    return self.hit() if action == HIT else self.stick()

  def compute_state(self, player_sum, usable_ace, dealer_card):
    return (usable_ace * N_POSSIBLE_PLAY_SUMS * N_DEAL_SCORES +
            N_POSSIBLE_PLAY_SUMS * (dealer_card - MIN_DEAL_CARD) +
            (player_sum - MIN_PLAY_SUM))

  def get_state(self):
      return self.compute_state(self.players['player'].sum,
                                self.players['player'].usable_ace,
                                self.players['dealer'].card_value(0))

  def decode_state(self, s):
    """Inverse of the function get_state but with state as input."""
    player_sum = (s % N_POSSIBLE_PLAY_SUMS) + MIN_PLAY_SUM
    s = (s - (player_sum - MIN_PLAY_SUM)) // N_POSSIBLE_PLAY_SUMS
    dealer_card = (s % N_DEAL_SCORES) + MIN_DEAL_CARD
    player_usable_ace = (s - (dealer_card - MIN_DEAL_CARD)) // N_DEAL_SCORES
    return player_sum, player_usable_ace, dealer_card

  def player_cards(self, player_sum, player_usable_ace):
    cards = []
    if player_usable_ace:
      cards.append(ACE_LOW)
      player_sum -= ACE_HIGH
    while player_sum > 0:
      card = min(player_sum, 10)
      cards.append(card)
      player_sum -= card
    return cards

  def force_state(self, s):
    """Forces initial state to be s (e.g. for exploring starts)."""
    player_sum, usable_ace, dealer_card = self.decode_state(s)
    self.players['player'].reset(self.player_cards(player_sum, usable_ace))
    self.players['dealer'].reset([dealer_card])

  def reset(self):
    for player in self.players.values():
      player.reset()
    return self.get_state()

  def _p(self, s_p, r, s, a):
    """Transition function defined in private because p dictionary in mdp.py."""
    pass

  def __str__(self):
    return (f"#####\nPLAYER: {self.players['player']}\n" +
            f"DEALER: {self.players['dealer']}")
