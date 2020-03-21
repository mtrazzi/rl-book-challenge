import argparse

from car_rental import CarRentalEnv
from dynamic_programming import DynamicProgramming
from gridworld import Gridworld


DEF_FIG_4_1_SIZE = 4
DEF_FIG_4_2_SIZE = 21


def random_policy(env):
  def pi(s, a):
    return 1 / len(env.moves)
  return pi


def fig_4_1(size=None):
  if size is None:
    size = DEF_FIG_4_1_SIZE
  env = Gridworld(size)
  pi_rand = random_policy(env)
  pi_init = {(a, s): pi_rand(s, a) for s in env.states for a in env.moves}
  alg = DynamicProgramming(env, pi=pi_init, theta=1e-4, gamma=1)  # undiscounted
  alg.policy_evaluation()
  alg.print_values()
  # show the optimal policy
  while not alg.policy_improvement():
    pass
  alg.print_policy_gridworld()


def fig_4_2(size=None):
  if size is None:
    size = DEF_FIG_4_2_SIZE
  # size - 1 because nb of cars from 0 to "size" param
  env = CarRentalEnv(size - 1)
  from utils import print_transitions, print_old_psums, print_psums
  print_psums(env)
  # print_transitions(env, print_zeros=True)
  # overflow_policy = {(0, 0): 0, (1, 0): 1, (0, 1): 0, (1, 1): 0}
  # subject_policy = {s: 0 for s in env.states}
  # alg = DynamicProgramming(env, pi=subject_policy, gamma=0.9, theta=1)
  # alg.policy_iteration()
  # alg.print_values(show_matplotlib=True)
  # alg.print_policy_car_rental()


PLOT_FUNCTION = {
  '4.1': fig_4_1,
  '4.2': fig_4_2,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=['4.1', '4.2'])
  parser.add_argument('-s', '--size', type=int, default=None,
                      help='Size of the environment (size * size states).')
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure](args.size)


if __name__ == "__main__":
  main()
