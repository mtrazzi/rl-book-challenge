from agents import RandomAgent, RLAgent
from utils import benchmark, opposite_agent, play_against_agent


def self_play(n_iterations=10, ben_steps=1000, training_steps=int(1e4),
              n_eval_episodes=100, **kwargs):
  """
  Returns an agent that learns from playing against himself from random to
  optimal play.
  """
  agents = [RLAgent(**kwargs), RandomAgent()]
  for _ in range(n_iterations):
    benchmark(agents[0], agents[1], ben_steps, training_steps, n_eval_episodes)
    # adding the trained agent as the new opponent to exploit
    agents[1] = opposite_agent(agents[0])
    agents[1].eps = agents[0].original_eps
  return agents[0]


def main():
  rl_agent_hyperparams = {"step": 0.1, "eps": 0.5, "eps_decay": 0.99}
  play_against_agent(self_play(n_iterations=4, training_steps=int(1e4),
                     **rl_agent_hyperparams))


if __name__ == '__main__':
  main()
