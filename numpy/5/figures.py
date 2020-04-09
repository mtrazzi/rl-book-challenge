import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from blackjack import BlackjackEnv, HIT, STICK, N_POSSIBLE_PLAY_SUMS, MIN_DEAL_CARD
from mc import (MonteCarloFirstVisit, MonteCarloES, OffPolicyMCControl,
                OffPolicyMCPrediction, OnPolicyFirstVisitMonteCarlo)
from one_state import LEFT, OneState, RIGHT, S_INIT
from racetrack import RacetrackEnv, Position, Velocity, RaceState

POLICY_THRESHOLD = 20
FIG_5_3_N_RUNS = 100
FIG_5_3_STATE_VALUE = -0.27726
FIG_5_3_PLAYER_SUM = 13
FIG_5_3_USABLE_ACE = True
FIG_5_3_DEALER_CARD = 2
FIG_5_3_MAX_EP = 10 ** 4
FIG_5_3_N_ESTIMATION_EP = 100000
FIG_5_4_N_RUNS = 10
FIG_5_4_MAX_EP = 10 ** 8
FIG_5_5_MAX_EP = 10 ** 2

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
      a = alg.sample_action(i, det=alg.det_pi is not None)
      to_print[player_sum - MIN_PLAY_SUM, dealer_card - MIN_DEAL_CARD] = a
  X = Y = list(range(to_print.shape[0]))
  Z = [[to_print[x, y] for y in Y] for x in X]
  dealer_idxs = np.arange(MIN_DEAL_CARD, MIN_DEAL_CARD + N_DEAL_SCORES)
  player_idxs = np.arange(MIN_PLAY_SUM, BLACKJACK + 1)
  sns.heatmap(Z, xticklabels=dealer_idxs, yticklabels=player_idxs,
              cbar_kws={'label': '0 = STICK, 1 = HIT'})
  ax.invert_yaxis()
  ax.set_title(title)

def print_race_policy(fig, alg):
  env = alg.env
  grid = env.race_map.grid
  pi = alg.det_target

  def print_speed_grid(pol, grid, axis, vel, fig_id):
    ax = fig.add_subplot('22' + str(fig_id))
    #ax.set_title(f'speed on axis = {"x" if axis == 0 else "y"}')
    to_print = np.zeros_like(grid)
    for x in range(grid.shape[0]):
      for y in range(grid.shape[1]):
        if grid[x, y]:
          pos = Position(x,y) 
          to_print[x,y] = (pi[RaceState(pos, vel)].x if axis == 0 else  pi[RaceState(pos, vel)].y)
    sns.heatmap(to_print, xticklabels=[], yticklabels=[])

  x_vel = Velocity(1, 0)
  y_vel = Velocity(0, 1) 
  for (idx, vel) in enumerate([x_vel, y_vel]):
    for axis in [0, 1]:
      print_speed_grid(pi, grid, axis, vel, idx * 2 + axis + 1)
  plt.show()

def random_policy(env):
  p_uniform = 1 / len(env.moves)
  return {(a, s): p_uniform for a in env.moves for s in env.states}


def blackjack_policy(env):
  def policy(s, a):
    player_sum, _, _ = env.decode_state(s)
    return float(a == (STICK if player_sum >= POLICY_THRESHOLD else HIT))
  return {(a, s): policy(s, a) for s in env.states for a in env.moves}


def blackjack_det_policy(env):
  def policy(s):
    player_sum, _, _ = env.decode_state(s)
    return STICK if player_sum >= POLICY_THRESHOLD else HIT
  return {s: policy(s) for s in env.states}

def generate_step_list(n_episodes):
  step_list = []
  step = 1
  base = 10
  while step < n_episodes:
    for i in range(1, base):
      step_list.append(i * step)
    step *= base
  return step_list + [n_episodes]

def fig_5_1():
  env = BlackjackEnv()
  fig = plt.figure()
  fig.suptitle('Figure 5.1')
  for (i, n_episodes) in enumerate([10000, 500000]):
    alg = MonteCarloFirstVisit(env, pi=blackjack_policy(env), gamma=1)
    alg.first_visit_mc_prediction(n_episodes=n_episodes)
    for (j, usable_ace) in enumerate([True, False]):
      fig_id = '22' + str(2 * j + i + 1)
      title = f"After {n_episodes} episodes "
      title += f"({'No usable' if not usable_ace else 'Usable'} ace)"
      print_plot(values_to_grid(env, alg.V, usable_ace=usable_ace),
                 title=title, fig=fig, fig_id=fig_id)
  plt.show()


def fig_5_2(n_episodes=int(1e5), on_policy_instead=False):
  env = BlackjackEnv()
  fig = plt.figure()
  fig.suptitle('Figure 5.2')
  if on_policy_instead:
    alg = OnPolicyFirstVisitMonteCarlo(env, pi=blackjack_policy(env),
                                       det_pi=None, gamma=1, epsilon=1e-2)
  else:
    alg = MonteCarloES(env, pi=blackjack_policy(env),
                       det_pi=blackjack_det_policy(env), gamma=1)
  alg.estimate_optimal_policy(n_episodes=n_episodes)
  alg.estimate_V_from_Q()
  for (j, usable_ace) in enumerate([True, False]):
    def fig_id(j, policy): return '22' + str(2 * (j + 1) - policy)
    title = f"({'No usable' if not usable_ace else 'Usable'} ace)"
    print_policy(alg, usable_ace, title, fig, fig_id(j, policy=True))
    print_plot(values_to_grid(env, alg.V, usable_ace=usable_ace),
               title='v*' + title, fig=fig, fig_id=fig_id(j, policy=False))
  plt.show()


def fig_5_3(n_episodes):
  n_episodes = FIG_5_3_MAX_EP if n_episodes == None else n_episodes
  env = BlackjackEnv()
  fig, ax = plt.subplots()
  plt.title('Figure 5.3')
  fig_5_3_state = env.compute_state(FIG_5_3_PLAYER_SUM, FIG_5_3_USABLE_ACE,
                                    FIG_5_3_DEALER_CARD)
  step_list = generate_step_list(n_episodes)

  # computing the value of the state from example 5.4 with first visit MC
  # to check if we get "-0.27726"
  # estimat_alg = MonteCarloFirstVisit(env, pi=blackjack_policy(env), gamma=1)
  # estimat_alg.first_visit_mc_prediction(FIG_5_3_N_ESTIMATION_EP, fig_5_3_state)
  # print(estimat_alg.V[fig_5_3_state])
  # -> result: turns out i get -0.28051 after 100000 episodes!

  def compute_errors(alg, step_list, start_state):
    errors = np.zeros(len(step_list))
    all_estimates = []
    for seed in range(FIG_5_3_N_RUNS):
      print(f"\n\n@@@@@@@@@@@\n\n RUN #{seed} \n\n@@@@@@@@@@@\n\n")
      alg.reset()
      estimates = alg.estimate_state(step_list, start_state, seed)
      all_estimates.append(estimates)
      errors = errors + (estimates - FIG_5_3_STATE_VALUE) ** 2
    return (1 / FIG_5_3_N_RUNS) * errors
  for weighted in [True, False]:
    label = ('Weighted' if weighted else 'Ordinary') + ' Importance Sampling'
    color = 'g' if not weighted else 'r'
    alg = OffPolicyMCPrediction(env, pi=blackjack_policy(env),
                                weighted=weighted, b=random_policy(env),
                                gamma=1)
    errors = compute_errors(alg, step_list, fig_5_3_state)
    plt.plot(step_list, errors, color=color, label=label)
  plt.xscale('log')
  ax.set_xticks(step_list)
  ax.set_xlabel('Episodes (log scale)')
  ax.set_ylabel(f'Mean square error (average over {FIG_5_3_N_RUNS} runs)')
  plt.legend()
  plt.show()

def fig_5_4(n_episodes):
  n_episodes = FIG_5_4_MAX_EP if n_episodes == None else n_episodes
  # plot initialization
  fig, ax = plt.subplots()
  plt.title('Figure 5.4')
  fig_5_4_state = S_INIT

  # algorithm initialization
  env = OneState()
  always_left_policy = {(a, s): float(a == LEFT) for a in env.moves for s in env.states}
  alg = OffPolicyMCPrediction(env, pi=always_left_policy,
                              weighted=False, b=random_policy(env),
                              gamma=1)

  # algorithm runs
  step_list = generate_step_list(n_episodes)
  for seed in range(FIG_5_4_N_RUNS):
    alg.reset()
    estimates = alg.estimate_state(step_list, fig_5_4_state, seed)
    plt.plot(step_list, estimates)

  # plotting
  const = np.zeros_like(step_list)
  for y in [1, 2]:
    plt.plot(step_list, const + y, color='black', linestyle='dashed')
  plt.xscale('log')
  ax.set_xticks(step_list)
  ax.set_xlabel('Episodes (log scale)')
  ax.set_yticks([0, 1, 2])
  ax.set_ylabel('MC estimate of v_pi(s) with ordinary ' + 
                f'import. samp. ({FIG_5_4_N_RUNS} runs)')
  plt.show()

def fig_5_5(n_episodes, config_file): 
  n_episodes = FIG_5_5_MAX_EP if n_episodes == None else n_episodes
  config_file = '1.txt' if config_file is None else config_file
  fig, ax = plt.subplots()
  plt.title('Figure 5.5')
  env = RacetrackEnv(config_file)
  env.seed(0)
  start_state = env.reset() 

  # runs
  step_list = generate_step_list(n_episodes)
  alg = OffPolicyMCControl(env, pi=random_policy(env),
                           b=random_policy(env),
                           gamma=1)
  alg.optimal_policy(n_episodes=n_episodes, start_state=start_state, step_list=step_list)
  print_race_policy(fig, alg)

PLOT_FUNCTION = {
  '5.1': fig_5_1,
  '5.2': fig_5_2,
  '5.3': fig_5_3,
  '5.4': fig_5_4, 
  '5.5': fig_5_5, 
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  parser.add_argument('-n', '--n_ep', type=int, default=None,
                      help='Number of episodes.')
  parser.add_argument('-o', '--on_policy_instead', type=bool, default=False,
                      help='For testing on-policy first visit MC control.')
  parser.add_argument('-c', '--config', type=str, default='configs/trivial.txt',
                      help='Config file for the maps of figure 5.5.')
  args = parser.parse_args()

  if args.figure in ['5.1']:
    PLOT_FUNCTION[args.figure](args.n_ep, args.on_policy_instead)
  elif args.figure in ['5.3', '5.4']:
    PLOT_FUNCTION[args.figure](args.n_ep)
  elif args.figure in ['5.5']:
    PLOT_FUNCTION[args.figure](args.n_ep, args.config)

if __name__ == "__main__":
  main()
