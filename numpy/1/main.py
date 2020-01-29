import copy

from agents import RandomAgent, RLAgent
from utils import benchmark, play_against_agent


def self_play(n_iterations=10, step=1000, training_steps=int(1e4),
              n_eval_episodes=100):
  """
  Returns an agent that learns from playing against himself from random to
  optimal play.
  """
  agents = [RLAgent(), RandomAgent()]
  for _ in range(n_iterations):
    benchmark(agents[0], agents[1], step, training_steps, n_eval_episodes)
    # adding the trained agent as the new opponent to exploit
    agents[1] = copy.deepcopy(agents[0])
  return agents[0]


def main():
  play_against_agent(self_play())


if __name__ == '__main__':
  main()
