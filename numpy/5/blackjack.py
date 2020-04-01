from mdp import MDP
import random

R_LOSE = -1
R_DRAW = 0
R_WIN = 1
R_STEP = 0
STICK = 0
HIT = 1
MIN_SUM = 12
BLACKJACK = 21
N_DEAL_SCORES = 10
ACE_STATES = 2
NUMBER_CARDS = 52
NUMBER_COLORS = 4
NB_VALUES = NUMBER_CARDS // NUMBER_COLORS
ACE_LOW = 1
ACE_HIGH = 11
ACE_DIFF = ACE_HIGH - ACE_LOW
N_CARD_DEALER = 1
N_CARD_PLAYER = 2
N_POSSIBLE_PLAY_SUMS = BLACKJACK - MIN_SUM + 1
DEALER_THRESHOLD = 17


class Player:
  def __init__(self, n_initial_cards):
    """Player starts with two, dealer with one hidden / one visible."""
    self.n_initial_cards = n_initial_cards
    self.cards = []
    self.sum = 0
    self.reset()
    self.bust = False

  def sample_card(self):
    return random.randint(ACE_LOW, NB_VALUES)

  def cards_to_values(self, card):
    """Replace the value of kings, queens, jacks to 10."""
    return min(card, 10)

  def new_card(self):
    print(f"new card begin (sum={self.sum})")
    self.cards.append(self.sample_card())
    print(f"added card {self.cards[-1]}")
    new_card_value = self.cards_to_values(self.cards[-1])
    # TODO: maybe some edge cases below with two aces etc.
    # if self.sum + new_card_value <= BLACKJACK:
    #   self.sum += self.cards_to_values(self.cards[-1])
    # else:
    #   self.bust = True
    self.sum += self.cards_to_values(self.cards[-1])
    self.bust = self.sum > BLACKJACK
    print(f"new card ends (sum={self.sum})")

  def update_aces(self):
    # TODO: maybe some edge cases below with two aces etc.
    self.usable_ace = (ACE_LOW in self.cards)
    if self.usable_ace and (self.sum + ACE_DIFF <= BLACKJACK):
      self.sum += ACE_DIFF

  def reset(self):
    self.cards = []
    self.sum = 0
    for _ in range(self.n_initial_cards):
      self.new_card()
    self.update_aces()

  def __str__(self):
    return (f"CARDS={self.cards}, " +
            f"SUM={self.sum}, " +
            f"USABLE_ACE={self.usable_ace}")


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
    return list(range((BLACKJACK - MIN_SUM + 1) * N_DEAL_SCORES * ACE_STATES))

  @property
  def r(self):
    return [R_LOSE, R_DRAW, R_WIN]

  def is_natural(self):
    return (self.players['player'].sum == BLACKJACK
            and len(self.players['player'].cards) == 2)

  def get_result(self):
    sum_diff = self.players["player"].sum - self.players["dealer"].sum
    if sum_diff > 0 or self.players['dealer'].bust:
      return R_WIN
    elif sum_diff == 0:
      return R_DRAW
    else:
      return R_LOSE

  def play_dealer(self):
    while self.players['dealer'].sum < DEALER_THRESHOLD:
      self.players['dealer'].new_card()

  def hit(self):
    s = self.get_state()
    self.players['player'].new_card()
    done = not self.players['player'].bust
    if done:
      return s, R_LOSE, done, {}  # returning s is arbitrary
    return self.get_state(), R_STEP, done, {}

  def stick(self):
    self.play_dealer()
    return self.get_state(), self.get_result(), True, {}

  def step(self, action):
    if self.is_natural():
      return self.get_state(), R_WIN, True, {}
    return self.hit() if action == HIT else self.stick()

  def compute_state(self, player_sum, usable_ace, dealer_card):
    # print(player_sum, usable_ace, dealer_card)
    return (usable_ace * N_POSSIBLE_PLAY_SUMS * N_DEAL_SCORES +
            N_POSSIBLE_PLAY_SUMS * (dealer_card - 1) +
            (player_sum - 1))

  def get_state(self):
    return self.compute_state(self.players['player'].sum,
                              self.players['player'].usable_ace,
                              self.players['dealer'].cards[0])

  def decode_state(self, s):
    """Inverse of the function get_state but with state as input."""
    player_sum = (s % N_POSSIBLE_PLAY_SUMS) + 1
    s = (s - (player_sum - 1)) // N_POSSIBLE_PLAY_SUMS
    dealer_card = (s % N_DEAL_SCORES) + 1
    player_usable_ace = (s - (dealer_card - 1)) // N_DEAL_SCORES
    return player_sum, player_usable_ace, dealer_card

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
