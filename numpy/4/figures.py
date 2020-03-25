import argparse

from car_rental import CarRentalEnv
from dynamic_programming import DynamicProgramming
from gridworld import Gridworld


DEF_FIG_4_1_SIZE = 4
DEF_FIG_4_2_SIZE = 21
DEF_EX_4_4_SIZE = 3


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
  alg.print_policy()


def fig_4_2(size=None):
  if size is None:
    size = DEF_FIG_4_2_SIZE
  # size - 1 because nb of cars from 0 to "size" param
  env = CarRentalEnv(size - 1)
  subject_policy = {s: 0 for s in env.states}
  alg = DynamicProgramming(env, det_pi=subject_policy, gamma=0.9, theta=1e-4)
  alg.policy_iteration()
  alg.print_values(show_matplotlib=True)
  alg.print_policy()


def ex_4_4(size=None):
  """
  Testing a policy iteration that stops when policy encountered twice on
  environment where all policies are equally bad (gridworld with cost of move
  equal to zero).
  """
  if size is None:
    size = DEF_EX_4_4_SIZE
  env = Gridworld(size, cost_move=0)
  det_pi = {s: env.moves[0] for s in env.states}
  alg = DynamicProgramming(env, det_pi=det_pi, theta=1e-7, gamma=1)
  # uncomment/comment for the difference between improvement and not improvement
  alg.policy_iteration_improved()
  # alg.policy_iteration()  # only converge if lucky fixed point


def ex_4_5(size=None):
  """
  Testing policy evaluation and policy iteration on gridworld using Q values.
  """
  if size is None:
    size = DEF_EX_4_4_SIZE
  env = Gridworld(size)
  pi_rand = random_policy(env)
  pi_init = {(a, s): pi_rand(s, a) for s in env.states for a in env.moves}
  alg = DynamicProgramming(env, pi=pi_init, theta=1e-4, gamma=1)
  alg.policy_iteration_Q()
  alg.print_policy()


def ex_4_7(size=None):
  if size is None:
    size = DEF_FIG_4_2_SIZE
  # size - 1 because nb of cars from 0 to "size" param
  env = CarRentalEnv(size - 1, ex_4_7=True)
  subject_policy = {s: 0 for s in env.states}
  alg = DynamicProgramming(env, det_pi=subject_policy, gamma=0.9, theta=1e-4)
  alg.policy_iteration()
  alg.print_values(show_matplotlib=True)
  alg.print_policy()


PLOT_FUNCTION = {
  '4.1': fig_4_1,
  '4.2': fig_4_2,
  'ex4.4': ex_4_4,
  'ex4.5': ex_4_5,
  'ex4.7': ex_4_7,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  parser.add_argument('-s', '--size', type=int, default=None,
                      help='Size of the environment (size * size states).')
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure](args.size)


if __name__ == "__main__":
  main()
