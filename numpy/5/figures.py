import argparse
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import seaborn as sns


from blackjack import BlackjackEnv
from mc import MonteCarloFirstVisit, MonteCarloES

STICK = 0
HIT = 1
POLICY_THRESHOLD = 20
MIN_PLAY_SUM = 12
MIN_DEAL_CARD = 1
BLACKJACK = 21
N_DEAL_SCORES = 10
N_POSSIBLE_PLAY_SUMS = BLACKJACK - MIN_PLAY_SUM + 1


def values_to_grid(env, V, usable_ace):
  """Puts values V into a printable grid form depending on usable_ace."""
  states = [env.decode_state(i) for i in V.keys()]
  to_print = np.zeros((N_POSSIBLE_PLAY_SUMS, N_DEAL_SCORES))
  for (i, (player_sum, usab, dealer_card)) in enumerate(states):
    if usab == usable_ace:
      to_print[player_sum - MIN_PLAY_SUM, dealer_card - MIN_DEAL_CARD] = V[i]
  return to_print


def print_plot(to_print, title, fig, fig_id):
  """Prints the grid `to_print` as presented in Figure 5.1. and 5.3."""
  dealer_idxs = np.arange(MIN_DEAL_CARD, MIN_DEAL_CARD + N_DEAL_SCORES)
  player_idxs = np.arange(MIN_PLAY_SUM, BLACKJACK + 1)
  ax = fig.add_subplot(fig_id, projection='3d')
  ax.set_title(title, fontsize=10)
  (X, Y), Z = np.meshgrid(dealer_idxs, player_idxs), to_print
  ax.set_xlabel('Dealer showing', fontsize=8)
  ax.set_ylabel('Player Sum', fontsize=8)
  ax.set_xticks([dealer_idxs.min(), dealer_idxs.max()])
  ax.set_yticks([player_idxs.min(), player_idxs.max()])
  ax.set_zticks([-1, 1])
  ax.plot_surface(X, Y, Z)


def print_policy(alg, usab_ace, title, fig, fig_id):
  ax = fig.add_subplot(fig_id)
  ax.set_title(title, fontsize=10)
  to_print = np.zeros((N_POSSIBLE_PLAY_SUMS, N_DEAL_SCORES))
  states = [alg.env.decode_state(i) for i in alg.V.keys()]
  for (i, (player_sum, usab, dealer_card)) in enumerate(states):
    if usab == usab_ace:
      a = alg.sample_action(i)
      to_print[player_sum - MIN_PLAY_SUM, dealer_card - MIN_DEAL_CARD] = a
  X = Y = list(range(to_print.shape[0]))
  Z = [[to_print[x, y] for y in Y] for x in X]
  dealer_idxs = np.arange(MIN_DEAL_CARD, MIN_DEAL_CARD + N_DEAL_SCORES)
  player_idxs = np.arange(MIN_PLAY_SUM, BLACKJACK + 1)
  sns.heatmap(Z, xticklabels=dealer_idxs, yticklabels=player_idxs,
              cbar_kws={'label': '0 = STICK, 1 = HIT'})
  ax.invert_yaxis()
  ax.set_title(title)


def blackjack_policy(env):
  def policy(s, a):
    player_sum, _, _ = env.decode_state(s)
    return a == (STICK if player_sum >= POLICY_THRESHOLD else HIT)
  return {(a, s): policy(s, a) for s in env.states for a in env.moves}


def blackjack_det_policy(env):
  def policy(s):
    player_sum, _, _ = env.decode_state(s)
    return STICK if player_sum >= POLICY_THRESHOLD else HIT
  return {s: policy(s) for s in env.states}


def fig_5_1():
  env = BlackjackEnv()
  fig = plt.figure()
  fig.suptitle('Figure 5.1')
  for (i, n_episodes) in enumerate([1000, 5000]):
    alg = MonteCarloFirstVisit(env, pi=blackjack_policy(env), gamma=1)
    alg.first_visit_mc_prediction(n_episodes=n_episodes)
    for (j, usable_ace) in enumerate([True, False]):
      fig_id = '22' + str(2 * j + i + 1)
      title = f"After {n_episodes} episodes "
      title += f"({'No usable' if not usable_ace else 'Usable'} ace)"
      print_plot(values_to_grid(env, alg.V, usable_ace=usable_ace),
                 title=title, fig=fig, fig_id=fig_id)
  plt.show()


def fig_5_3(n_episodes=int(1e5)):
  env = BlackjackEnv()
  fig = plt.figure()
  fig.suptitle('Figure 5.3')
  alg = MonteCarloES(env, pi=blackjack_policy(env), det_pi=blackjack_det_policy(env), gamma=1)
  alg.estimate_optimal_policy(n_episodes=n_episodes)
  alg.estimate_V_from_Q()
  for (j, usable_ace) in enumerate([True, False]):
    def fig_id(j, policy): return '22' + str(2 * (j + 1) - policy)
    title = f"({'No usable' if not usable_ace else 'Usable'} ace)"
    print_policy(alg, usable_ace, title, fig, fig_id(j, policy=True))
    print_plot(values_to_grid(env, alg.V, usable_ace=usable_ace),
               title='v*' + title, fig=fig, fig_id=fig_id(j, policy=False))
  plt.show()


PLOT_FUNCTION = {
  '5.1': fig_5_1,
  '5.3': fig_5_3,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  parser.add_argument('-n', '--n_ep', type=int, default=None,
                      help='Number of episodes.')
  args = parser.parse_args()

  if args.figure in ['5.3']:
    PLOT_FUNCTION[args.figure](args.n_ep)
  else:
    PLOT_FUNCTION[args.figure]()


if __name__ == "__main__":
  main()
