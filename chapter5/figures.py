import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time


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
FIG_5_5_MAX_PRINT_VEL = 1
FIG_5_5_BLACK_COLOR = -1
FIG_5_5_FINISH_COLOR = 0
FIG_5_5_INIT_COLOR = 0
FIG_5_5_N_INTERM_TRAJS = 100
INITIAL_STATE_IDX = 0

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

def print_race_policy(alg):
  env = alg.env
  grid = env.race_map.grid
  pi = alg.det_target

  def print_speed_grid(fig, pol, grid, axis, vel, fig_id, fig_id_base):
    ax = fig.add_subplot(str(fig_id_base) + str(fig_id))
    ax.set_title(f'axis = {"x" if axis == 0 else "y"}, vel = {str(vel)}')
    to_print = np.zeros_like(grid) - 2
    for x in range(grid.shape[0]):
      for y in range(grid.shape[1]):
        pos = Position(x,y) 
        s = RaceState(pos, vel)
        if grid[x, y] and s.is_valid(env.race_map):
          a_best = pi[s]
          to_print[x,y] = (a_best.x if axis == 0 else a_best.y)
    sns.heatmap(to_print, xticklabels=[], yticklabels=[])
    ax.invert_yaxis()

  
  x_vels = [Velocity(x, 0) for x in range(FIG_5_5_MAX_PRINT_VEL + 1)]
  y_vels = [Velocity(0, y) for y in range(FIG_5_5_MAX_PRINT_VEL + 1)]
  for (idx, vel) in enumerate(x_vels + y_vels):
    fig = plt.figure()
    for axis in [0, 1]:
      print_speed_grid(fig, pi, grid, axis, vel, axis + 1, '12')
    plt.show()


def plot_race_traj(alg, start_state, debug=True, max_steps=np.inf, eps=None, total_ep=None, title_fig='Fig 5.5'):
  # generating trajectories
  alg.det_pi = alg.det_target
  traj = alg.generate_trajectory(start_state=start_state, det=True, max_steps=max_steps, eps=eps)
  
  # initial map coloring
  race_map = alg.env.race_map
  color_grid = copy.copy(race_map.grid)
  mask = copy.copy(1-race_map.grid)
  for pos in race_map.finish_line:
    color_grid[pos.x, pos.y] = FIG_5_5_FINISH_COLOR
  for s_init in race_map.initial_states:
    color_grid[s_init.p.x, s_init.p.y] = FIG_5_5_INIT_COLOR
  backup_grid = copy.copy(color_grid) 

  def color_traj(s, a, color=None):
    x, y = s.p.x, s.p.y
    delta_x, delta_y = s.v.x + a.x, s.v.y + a.y
    dx, dy = np.sign(delta_x), np.sign(delta_y) 
    color = color if color is not None else backup_grid[x, y]
    while True:
      color_grid[x, y] = color
      if abs(delta_x) != abs(delta_y):   
        if ((abs(x - s.p.x) + 1) / (abs(y - s.p.y) + 1)) > ((abs(delta_x) + 1) / (abs(delta_y) + 1)):
          y += dy
        else:
          x += dx
      else:
        x, y = x + dx, y + dy
      if (x == (s.p.x + delta_x) and y == (s.p.y + delta_y)) or not (0 <= x < color_grid.shape[0]) or not (0 <= y < color_grid.shape[1]):
        if (0 <= x < color_grid.shape[0]) and (0 <= y < color_grid.shape[1]):
          color_grid[x, y] = color
        break

  # plotting 
  def show_reverse(grid, mask_arr):
      sns.heatmap(grid[::-1], mask=mask_arr[::-1],cbar_kws={'label': 'velocity norm'})
      plt.show()
 
  fig, ax = plt.subplots()
  nb_ep = '' if total_ep is None else f"after {total_ep} episodes: "
  ax.set_title(f"{title_fig} - speed heatmap - {nb_ep} optimal traj. takes {len(traj)} actions")
  ax.invert_xaxis()
  for (s, a, _) in traj:
    color = (s.v + a).norm()
    color_traj(s, a, color) 
    if debug:
      show_reverse(color_grid, mask)
      color_traj(s, a) 
  if not debug:
    show_reverse(color_grid, mask)

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

def fig_5_5(n_episodes, config_file, truncated_weighted_avg_est=False, title_fig='Fig 5.5'): 
  n_episodes = FIG_5_5_MAX_EP if n_episodes == None else n_episodes
  config_file = '1.txt' if config_file is None else config_file
  env = RacetrackEnv(config_file)
  gamma = 0.9 if truncated_weighted_avg_est else 1
  for start_state in env.race_map.initial_states[INITIAL_STATE_IDX:]:
    # training runs
    env.seed(0)
    env.noise = True
    step_list = generate_step_list(n_episodes)
    alg = OffPolicyMCControl(env, pi=random_policy(env),
                             b=random_policy(env),
                             gamma=gamma)
 
    interm_trajs_length = n_episodes // FIG_5_5_N_INTERM_TRAJS
    total_ep = 0
    for i in range(FIG_5_5_N_INTERM_TRAJS):
      alg.env.noise = True
      optimisation_alg = alg.optimal_policy if not truncated_weighted_avg_est else alg.truncated_weighted_avg_est
      optimisation_alg(n_episodes=interm_trajs_length, start_state=start_state, step_list=step_list)
      total_ep += interm_trajs_length
      alg.env.noise = False
      plot_race_traj(alg, start_state, debug=False, max_steps=1000, total_ep=total_ep, eps=None, title_fig=title_fig)

def ex_5_14(n_episodes, config_file): 
  fig_5_5(n_episodes, config_file, truncated_weighted_avg_est=True, title_fig='Ex 5.14')

PLOT_FUNCTION = {
  '5.1': fig_5_1,
  '5.2': fig_5_2,
  '5.3': fig_5_3,
  '5.4': fig_5_4, 
  '5.5': fig_5_5, 
  'ex5.14': ex_5_14,
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
  elif args.figure in ['5.5', 'ex5.14']:
    PLOT_FUNCTION[args.figure](args.n_ep, args.config)

if __name__ == "__main__":
  main()
