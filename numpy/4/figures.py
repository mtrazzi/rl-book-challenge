import argparse

import numpy as np
from gridworld import Gridworld
from dynamic_programming import DynamicProgramming


def random_policy(env):
  def pi(s, a):
    return 1 / len(env.moves)
  return pi


def fig_4_1():
  env = Gridworld()
  alg = DynamicProgramming(env)
  alg.policy_evaluation(random_policy(env))


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
