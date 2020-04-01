import argparse

from blackjack import BlackjackEnv
from mc import MonteCarloFirstVisit

STICK = 0
HIT = 1
POLICY_THRESHOLD = 20


def blackjack_policy(env):
  def policy(s, a):
    player_sum, _, _ = env.decode_state(s)
    return a == (STICK if player_sum >= POLICY_THRESHOLD else HIT)
  return {(a, s): policy(s, a) for s in env.states for a in env.moves}


def fig_5_1(size=None):
  env = BlackjackEnv()
  alg = MonteCarloFirstVisit(env, pi=blackjack_policy(env), gamma=1)
  alg.first_visit_mc_prediction()
  # alg.print_values()


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
