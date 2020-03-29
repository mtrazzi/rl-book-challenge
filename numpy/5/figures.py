import argparse

from blackjack import BlackjackEnv
from mc import MonteCarloFirstVisit


def random_policy(env):
  def pi(s, a):
    return 1 / len(env.moves)
  return pi


def fig_5_1(size=None):
  env = BlackjackEnv()
  pi_rand = random_policy(env)
  pi_init = {(a, s): pi_rand(s, a) for s in env.states for a in env.moves}
  alg = MonteCarloFirstVisit(env, pi=pi_init, gamma=0.9)
  alg.first_visit_mc_prediction()
  alg.print_values()


PLOT_FUNCTION = {
  '5.1': fig_5_1,
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
