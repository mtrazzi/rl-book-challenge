GO_HOME = 0
STATES = ["leaving office",
          "reach car",
          "exiting highway",
          "2ndary road",
          "home street",
          "arrive home"]
TRAVEL_TIME = [5, 15, 10, 10, 3, 0]

class DrivingEnv:
  def __init__(self):
    self.reset()

  @property
  def moves(self):
    return [GO_HOME]

  @property
  def states(self):
    return STATES

  def associated_reward(self, state):
    return TRAVEL_TIME[self.states.index(state)]

  def step(self, action):
    state_idx = self.states.index(self.state) 
    done = state_idx == len(self.states) - 2
    new_state = self.states[(state_idx + 1) % len(self.states)]
    self.state = new_state
    return new_state, TRAVEL_TIME[state_idx], done, {}

  def reset(self):
    self.state = self.states[0]
    return self.state

  def __str__(self):
    return self.state
