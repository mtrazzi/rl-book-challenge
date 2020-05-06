import argparse
import matplotlib.pyplot as plt
from dyna_q import DynaQ
from dyna_maze import DynaMaze 
from models import FullModel
from tabular_q import TabularQ
from utils import sample, to_arr
import seaborn as sns
import numpy as np
from nstep_sarsa import nStepSarsa

SEC_8_1_ALP = 0.001
SEC_8_1_N_STEPS = int(1e6)
DYNA_MAZE_GAMMA = 0.95
FIG_8_2_ALP = 0.1
FIG_8_2_N_EP = 50
FIG_8_2_EPS = 0.1
FIG_8_2_PLAN_STEPS = [0, 5, 50]
FIG_8_2_C_DIC = {0: 'b', 5: 'g', 50: 'r'}
FIG_8_2_N_RUNS = 30
FIG_8_3_PLAN_STEPS = [0, 50]
FIG_8_3_HEAT_LAB = {(0, -1): 'left', (0, 1): 'right', (-1, 0): 'up', (1, 0): 'down'}
MED_FONT = 10
BIG_FONT = 15
EX_8_1_N_LIST = [5, 50]
FIG_8_4_INIT_POS = (5, 3)
FIG_8_4_GOAL_POS = (0, 8)
FIG_8_4_WALLS = [(3, y) for y in range(9)]
FIG_8_4_CHG_T = 1000
FIG_8_4_FINAL_T = 3000
FIG_8_4_PLAN_STEPS = 50
FIG_8_4_N_RUNS = 1
FIG_8_5_CHG_T = 3000

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def section_8_1():
  env = DynaMaze(FIG_8_2_INIT_POS, FIG_8_2_GOAL_POS, FIG_8_2_GRID_SHAPE, FIG_8_2_WALL)
  alg = TabularQ(FullModel(env), SEC_8_1_ALP, DYNA_MAZE_GAMMA)
  alg.seed(0)
  alg.rand_sam_one_step_pla(SEC_8_1_N_STEPS, decay=True)
  V = alg.get_V()
  plt.title('Section 8.1 - tabular Q (1-step random sample, dyna maze)')
  sns.heatmap(to_arr(V), cbar_kws={'label': 'max(Q(s, a))'})
  save_plot('section8.1')
  plt.show()

def fig_8_2():
  fig, ax = plt.subplots()
  env = DynaMaze()
  alg = DynaQ(env, FIG_8_2_ALP, DYNA_MAZE_GAMMA, FIG_8_2_EPS)
  xticks = [2, 10, 20, 30, 40, 50]
  yticks = [14, 200, 400, 600, 800]
  ax.set_title('Figure 8.2', fontsize=BIG_FONT)
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([0, max(yticks)])
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel('Episodes', fontsize=BIG_FONT)
  ax.set_ylabel('Steps\nper\nepisode', rotation=0, labelpad=25, fontsize=BIG_FONT)
  ep_ticks = list(range(2, 51))
  alg.seed(0)
  for n_plan_steps in FIG_8_2_PLAN_STEPS:
    arr_sum = np.zeros(FIG_8_2_N_EP)
    for _ in range(FIG_8_2_N_RUNS):
      alg.reset()
      arr_sum += np.array(alg.tabular_dyna_q(FIG_8_2_N_EP, n_plan_steps))
    plt.plot(ep_ticks, (arr_sum / FIG_8_2_N_RUNS)[1:], label=f'{n_plan_steps} planning steps', color=FIG_8_2_C_DIC[n_plan_steps])
  plt.legend()
  fig.set_size_inches(10, 8)
  save_plot('fig8.2', dpi=100)
  plt.show()

def get_dyna_maze_pol(env, Q):
  pi = {}
  for s in env.states:
    q_arr = np.array([Q[(s, a)] for a in env.moves_d[s]])
    is_max = q_arr == q_arr.max()
    arg_max_idx = np.flatnonzero(is_max)
    if np.all(is_max):
      pi[s] = 0
    else:
      pi[s] = arg_max_idx + 1
  return pi

def fig_8_3():
  fig = plt.figure()
  fig.suptitle('Figure 8.3 - Policies found by Dyna-Q after 2 episodes', fontsize=BIG_FONT)
  env = DynaMaze()
  alg = DynaQ(env, FIG_8_2_ALP, DYNA_MAZE_GAMMA, FIG_8_2_EPS)
  action_dict = {move_id: env.moves[move_id - 1] for move_id in range(1, len(env.moves) + 1)}
  heatmap_label = '0 = all equal'
  for move_id, move in action_dict.items():
    heatmap_label += f', {move_id}: {FIG_8_3_HEAT_LAB[move]}'
  for (i, n_plan_steps) in enumerate(FIG_8_3_PLAN_STEPS): 
    alg.seed(0)
    alg.reset()
    ax = fig.add_subplot(f'12{i + 1}')
    ax.set_title(f"with{'' if n_plan_steps > 0 else 'out'} planning (n={n_plan_steps})")
    alg.tabular_dyna_q(2, n_plan_steps)
    sns.heatmap(to_arr(get_dyna_maze_pol(env, alg.Q)), cbar_kws={'label': heatmap_label if i == 0 else None}, xticklabels=False, yticklabels=False)
  fig.set_size_inches(10, 8)
  save_plot('fig8.3', dpi=100)
  plt.show()

def ex_8_1():
  fig = plt.figure()
  fig.suptitle('Exercise 8.1 - Policies found by n-step sarsa after 2 episodes', fontsize=BIG_FONT)
  env = DynaMaze()
  action_dict = {move_id: env.moves[move_id - 1] for move_id in range(1, len(env.moves) + 1)}
  heatmap_label = '0 = all equal'
  for move_id, move in action_dict.items():
    heatmap_label += f', {move_id}: {FIG_8_3_HEAT_LAB[move]}'
  for (i, n) in enumerate(EX_8_1_N_LIST): 
    alg = nStepSarsa(env, step_size=FIG_8_2_ALP, gamma=DYNA_MAZE_GAMMA, n=n)
    alg.seed(0)
    alg.reset()
    ax = fig.add_subplot(f'12{i + 1}')
    ax.set_title(f'n={n}')
    alg.pol_eval(n_ep=2)
    sns.heatmap(to_arr(get_dyna_maze_pol(env, alg.Q)), cbar_kws={'label': heatmap_label if i == 0 else None}, xticklabels=False, yticklabels=False)
  fig.set_size_inches(10, 8)
  save_plot('ex8.1', dpi=100)
  plt.show()

def fig_8_4():
  fig, ax = plt.subplots()
  ax.set_title('Figure 8.4', fontsize=MED_FONT)
  xticks = [0, 1000, 2000, 3000]
  yticks = [0, 150]
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_ylim([min(yticks), max(yticks)])
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel('Time Steps', fontsize=MED_FONT)
  ax.set_ylabel('Cumulative\nReward', rotation=0, fontsize=MED_FONT)
  env = DynaMaze(FIG_8_4_INIT_POS, FIG_8_4_GOAL_POS, walls=FIG_8_4_WALLS[:-1])
  alg = DynaQ(env, FIG_8_2_ALP, DYNA_MAZE_GAMMA, FIG_8_2_EPS)
  alg.seed(0)
  arr_sum = np.zeros(FIG_8_4_FINAL_T)
  for run in range(FIG_8_4_N_RUNS):
    alg.reset()
    print(f"run {run + 1}/{FIG_8_4_N_RUNS}")
    alg.env = DynaMaze(FIG_8_4_INIT_POS, FIG_8_4_GOAL_POS, walls=FIG_8_4_WALLS[:-1])
    cum_rew_l_left = np.array(alg.tabular_dyna_q_step(FIG_8_4_CHG_T, FIG_8_4_PLAN_STEPS))
    alg.env = DynaMaze(FIG_8_4_INIT_POS, FIG_8_4_GOAL_POS, walls=FIG_8_4_WALLS[1:])
    cum_rew_l_right = np.array(alg.tabular_dyna_q_step(FIG_8_4_FINAL_T - FIG_8_4_CHG_T, FIG_8_4_PLAN_STEPS)) + cum_rew_l_left.max()
    arr_sum += np.array(list(cum_rew_l_left) + list(cum_rew_l_right))
  plt.plot(arr_sum / FIG_8_4_N_RUNS, label='Dyna-Q', color='b')
  fig.set_size_inches(10, 8)
  save_plot('8.4', dpi=100)
  plt.show()

PLOT_FUNCTION = {
  'section8.1': section_8_1,
  '8.2': fig_8_2,
  '8.3': fig_8_3, 
  '8.4': fig_8_4, 
  'ex8.1': ex_8_1,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=list(PLOT_FUNCTION.keys()) + ['all'])
  args = parser.parse_args()
  if args.figure == 'all':
    for key, f in PLOT_FUNCTION.items():
      print(f"[{key}]")
      f()
  else:
    print(f"[{args.figure}]")
    PLOT_FUNCTION[args.figure]()

if __name__ == '__main__':
  main()
