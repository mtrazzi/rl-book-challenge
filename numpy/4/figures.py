import argparse

import numpy as np
from gridworld import Gridworld
from dynamic_programming import DynamicProgramming
from utils import print_transitions


def random_policy(env):
  def pi(s, a):
    return 1 / len(env.moves)
  return pi


def fig_4_1():
  env = Gridworld()
  print_transitions(env)
  pi_rand = random_policy(env)
  pi_init = {(a, s): pi_rand(s, a) for s in env.states for a in env.moves}
  alg = DynamicProgramming(env, pi=pi_init, theta=0.001, gamma=1)  # undiscounted
  alg.policy_evaluation()
  alg.print_values()
  # show the optimal policy
  while not alg.policy_improvement(): pass
  alg.print_policy()




PLOT_FUNCTION = {
  '4.1': fig_4_1,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=['4.1'])
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure]()


if __name__ == "__main__":
  main()
