import argparse
import matplotlib.pyplot as plt
from dyna_q import DynaQ
from dyna_q_plus import DynaQPlus
from dyna_maze import DynaMaze
from dyna_maze_part import DynaMazePartitioned
from models import FullModel
from tabular_q import TabularQ
from utils import sample, to_arr
import seaborn as sns
import numpy as np
from nstep_sarsa import nStepSarsa
from prior_sweep import PrioritizedSweeping
from traj_sampling import TrajectorySampling
from task import Task, START_STATE
import time

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
MED_FONT = 13
BIG_FONT = 15
EX_8_1_N_LIST = [5, 50]
FIG_8_4_INIT_POS = (5, 3)
FIG_8_4_GOAL_POS_L = [(0, 8)]
FIG_8_4_GRID_SHAPE = (6, 9)
FIG_8_4_WALLS = [(3, y) for y in range(FIG_8_4_GRID_SHAPE[1])]
FIG_8_4_CHG_T = 1000
FIG_8_4_FINAL_T = 3000
FIG_8_4_PLAN_STEPS = 50
FIG_8_4_N_RUNS = 5
FIG_8_4_ALP = 0.5
FIG_8_4_EPS = 0.1
FIG_8_4_K = 0.001
FIG_8_5_CHG_T = 3000
FIG_8_5_FINAL_T = 6000
EX_8_4_CHG_T = 6000
EX_8_4_FINAL_T = 12000
EXAMPLE_8_4_THETA = 1e-4
EXAMPLE_8_4_N_PART = list(range(int(np.log(6016 // 47) / np.log(2)) + 1))
EXAMPLE_8_4_N_RUNS = 4
FIG_8_7_B_L = [2, 10, 100, 1000, 10000]
FIG_8_7_N_RUNS = 100
FIG_8_8_S_0 = START_STATE
FIG_8_8_N_ST_LOWER = 10000
FIG_8_8_N_UPD_LOWER = 200000
FIG_8_8_B_L_LOWER = [1]
FIG_8_8_LOG_FREQ_LOWER = 10000
FIG_8_8_B_L_UPPER = [1, 3, 10]
FIG_8_8_N_ST_UPPER = 1000
FIG_8_8_N_UPD_UPPER = 20000
FIG_8_8_LOG_FREQ_UPPER = 1000
FIG_8_8_N_RUNS = 10

def save_plot(filename, dpi=None):
  plt.savefig('plots/' + filename + '.png', dpi=dpi)

def show_pol(alg, show_label=True):
  action_dict = {move_id: alg.env.moves[move_id - 1] for move_id in range(1, len(alg.env.moves) + 1)}
  heatmap_label = '0 = all equal'
  for move_id, move in action_dict.items():
    heatmap_label += f', {move_id}: {FIG_8_3_HEAT_LAB[move]}'
  sns.heatmap(to_arr(get_dyna_maze_pol(alg.env, alg.Q)), cbar_kws={'label': heatmap_label if show_label else None}, xticklabels=False, yticklabels=False)

def section_8_1():
  env = DynaMaze(FIG_8_2_INIT_POS, FIG_8_2_GOAL_POS_L, FIG_8_2_GRID_SHAPE, FIG_8_2_WALL)
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
    if (s, env.moves_d[s][0]) not in Q:
      pi[s] = 0
      continue
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
  for (i, n) in enumerate(EX_8_1_N_LIST): 
    alg = nStepSarsa(env, step_size=FIG_8_2_ALP, gamma=DYNA_MAZE_GAMMA, n=n)
    alg.seed(0)
    alg.reset()
    ax = fig.add_subplot(f'12{i + 1}')
    ax.set_title(f'n={n}')
    alg.pol_eval(n_ep=2)
    show_pol(alg, i==0)
  fig.set_size_inches(10, 8)
  save_plot('ex8.1', dpi=100)
  plt.show()

def run_dynaq_dynaqp(title, filename, n_runs, xticks, yticks, change_t, final_t, walls1, walls2, plan_steps, alpha, eps, k, ex_8_4=False): 
  
  # initialization
  fig, ax = plt.subplots()
  env = DynaMaze(FIG_8_4_INIT_POS, FIG_8_4_GOAL_POS, FIG_8_4_GRID_SHAPE, walls1, walls2)
  dyna_q_alg = DynaQ(env, alpha, DYNA_MAZE_GAMMA, eps) if not ex_8_4 else DynaQPlus(env, alpha, DYNA_MAZE_GAMMA, eps, k)
  dyna_qp_alg = DynaQPlus(env, alpha, DYNA_MAZE_GAMMA, eps, k)
  for (alg, label) in [(dyna_q_alg, 'Dyna-Q' + '+ex8.4' * ex_8_4), (dyna_qp_alg, 'Dyna-Q+')]:
    arr_sum = np.zeros(final_t)
    alg.seed(0)
    for run in range(n_runs):
      alg.reset()
      opt = alg.ex_8_4 if (ex_8_4 and label[-1] == '4') else alg.tabular_dyna_q_step
      print(f"run {run + 1}/{n_runs}")
      cum_rew_l_left = opt(change_t, plan_steps)
      alg.env.switch_walls()
      cum_rew_l_right = np.array(opt(final_t - change_t, plan_steps)) + cum_rew_l_left[-1]
      arr_sum += np.array(cum_rew_l_left + list(cum_rew_l_right))
      alg.env.switch_walls()
    plt.plot(arr_sum / n_runs, label=label)
  
  # plot 
  plt.legend()
  ax.set_title(title + f" ({n_runs} runs average, n={plan_steps} planning steps, a={alpha}, k={k})", fontsize=MED_FONT)
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xlabel('Time Steps', fontsize=BIG_FONT)
  ax.set_ylabel('Cumulative\nReward', rotation=0, labelpad=15, fontsize=BIG_FONT-2)
  fig.set_size_inches(10, 8)
  save_plot(filename, dpi=100)
  plt.show()

def fig_8_4():
  run_dynaq_dynaqp('Figure 8.4', 'fig8.4', FIG_8_4_N_RUNS, [0, 1000, 2000, 3000], [0, 150], FIG_8_4_CHG_T, FIG_8_4_FINAL_T, FIG_8_4_WALLS[:-1], FIG_8_4_WALLS[1:], FIG_8_4_PLAN_STEPS, alpha=FIG_8_4_ALP, eps=FIG_8_4_EPS, k=FIG_8_4_K)

def fig_8_5():
  run_dynaq_dynaqp('Figure 8.5', 'fig8.5', FIG_8_4_N_RUNS, [0, 3000, 6000], [0, 400], FIG_8_5_CHG_T, FIG_8_5_FINAL_T, FIG_8_4_WALLS[1:], FIG_8_4_WALLS[1:-1], FIG_8_4_PLAN_STEPS, alpha=FIG_8_4_ALP, eps=FIG_8_4_EPS, k=FIG_8_4_K)

def ex_8_4():
  run_dynaq_dynaqp('Exercise 8.4', 'ex8.4', FIG_8_4_N_RUNS, [0, 6000, 12000], [0, 1000], EX_8_4_CHG_T, EX_8_4_FINAL_T, FIG_8_4_WALLS[1:], FIG_8_4_WALLS[:-1], FIG_8_4_PLAN_STEPS, alpha=FIG_8_4_ALP, eps=FIG_8_4_EPS, k=FIG_8_4_K, ex_8_4=True)

def example_8_4():
  fig, ax = plt.subplots()  
  ax.set_title(f'Example 8.4 ({EXAMPLE_8_4_N_RUNS} runs per datapoint)', fontsize=BIG_FONT)
  n_upd_prio_l, n_upd_dyna_l, n_states_l = [], [], []
  for n in EXAMPLE_8_4_N_PART:
    env = DynaMazePartitioned(n)
    n_moves_opt = sum(env.expand((6, 8)))
    n_states_l.append(len(env.states)-len(env.walls))
    print(f"n_states={n_states_l[-1]}")
    for (alg_class, param, n_upd_l) in [(PrioritizedSweeping, EXAMPLE_8_4_THETA, n_upd_prio_l), (DynaQ, FIG_8_4_EPS, n_upd_dyna_l)]:
      print(("prio" if alg_class == PrioritizedSweeping else "dyna") + "...")
      n_upd_l.append(0)
      alg = alg_class(env, FIG_8_4_ALP, DYNA_MAZE_GAMMA, param)
      alg.seed(0)
      for run in range(EXAMPLE_8_4_N_RUNS):
        alg.reset()
        start = time.time()
        n_upd_l[-1] += (alg.updates_until_optimal(n_moves_opt, n_plan_steps=5, tol=0.5))
        print(f"run #{run} took {time.time()-start:.2f}s")
      n_upd_l[-1] /= EXAMPLE_8_4_N_RUNS
  ax.set_xlabel('Gridworld states (#states)', fontsize=BIG_FONT-2)
  ax.set_ylabel('Updates\nuntil\noptimal\nsolution', rotation=0, labelpad=25, fontsize=BIG_FONT-2)
  ax.set_xscale('log', basex=2)
  x_name = ['0'] + [str(n_states) for n_states in n_states_l]
  xticks = [2 ** k for k in range(len(n_states_l) + 1)]
  plt.xticks(xticks, x_name)
  ax.set_yscale('log')
  plt.plot(xticks, [10] + n_upd_prio_l, color='r', label='Prioritized Sweeping')
  plt.plot(xticks, [10] + n_upd_dyna_l, color='b', label='Dyna-Q')
  plt.legend()
  fig.set_size_inches(10, 8)
  save_plot('example8.4', dpi=100)
  plt.show()

def fig_8_7():
  fig, ax = plt.subplots()  
  ax.set_title(f'Figure 8.7 ({FIG_8_7_N_RUNS} runs per b)', fontsize=BIG_FONT)
  R = 0
  gamma = 1
  def rms_error(vals, true_vals, n):
    return np.sqrt(np.sum((vals - true_vals) ** 2) / n)
  for b in FIG_8_7_B_L:
    print(f"b={b}")
    true_q_vals = np.random.randn(b)
    qstar = R + gamma * np.mean(true_q_vals)
    estim = np.zeros((2 * b + 1, FIG_8_7_N_RUNS))
    qhat_0 = qstar + np.random.choice([-1, 1])
    for run in range(FIG_8_7_N_RUNS):
      qhat = qhat_0
      errors = []
      for t in range(1, 2 * b + 1):
        sampled_idx = np.random.randint(b)
        qhat += (1 / (t + 1)) * (R + gamma * true_q_vals[sampled_idx] - qhat)
        estim[t - 1, run] = qhat
    xidxs = [x / (2 * b) for x in range(1, 2 * b + 1)]
    rms_err = [rms_error(estim[t - 1, :], qstar, FIG_8_7_N_RUNS) for t in range(1, 2 * b + 1)]
    plt.plot(xidxs,  rms_err, label=f'b={b}')
  xname = ['0', '1b', '2b']
  xticks = [0, 1 / 2, 1]
  plt.xticks(xticks, xname)
  plt.legend()
  save_plot('fig8.7')
  plt.show()

def set_axis(ax, n_states, xticks, show_ylabel=True):
  ax.set_title(f'{n_states} states')
  xlabel = 'Computation time, in expected updates'
  ylabel = 'Value of\nstart state\nunder\ngreedy\npolicy'
  ax.set_xlabel(xlabel, fontsize=BIG_FONT-4)
  if show_ylabel:
    ax.set_ylabel(ylabel, rotation=0, labelpad=35, fontsize=BIG_FONT-4)
  ax.set_xticks([xticks[k] for k in [0, 5, 10, 15, 20]])

def fig_8_8():
  fig = plt.figure() 
  np.random.seed(0)
  for (n_st, log_freq, n_upd, b_list, fig_id) in [(FIG_8_8_N_ST_UPPER, FIG_8_8_LOG_FREQ_UPPER, FIG_8_8_N_UPD_UPPER, FIG_8_8_B_L_UPPER, '121'),
                                                  (FIG_8_8_N_ST_LOWER, FIG_8_8_LOG_FREQ_LOWER, FIG_8_8_N_UPD_LOWER, FIG_8_8_B_L_LOWER, '122')]:
    xticks = [log_freq * k for k in range(n_upd // log_freq + 1)]
    set_axis(fig.add_subplot(fig_id), n_st, xticks + [n_upd], fig_id == '121')
    for b in b_list:
      task_list = [Task(b, n_st) for _ in range(FIG_8_8_N_RUNS)]
      print(f"b={b}")
      for label in ['uniform', 'on policy']:
        print(f"{label}..")
        vals = 0
        for (run_id, task) in enumerate(task_list):
          print(f"run #{run_id}")
          alg = TrajectorySampling(task) 
          updates = alg.uniform if label == 'uniform' else alg.on_policy
          vals += updates(FIG_8_8_S_0, n_upd, log_freq)
          print(vals)
        plt.plot(xticks,
                 [0] + list(vals / FIG_8_8_N_RUNS),
                 label=f'b={b}, ' + label) 
      plt.legend()
  fig.suptitle(f'Figure 8.8 ({FIG_8_8_N_RUNS} sample tasks)')
  fig.set_size_inches(20, 16)
  save_plot('fig8.8')
  plt.show()

PLOT_FUNCTION = {
  'section8.1': section_8_1,
  '8.2': fig_8_2,
  '8.3': fig_8_3, 
  '8.4': fig_8_4, 
  '8.5': fig_8_5, 
  'ex8.1': ex_8_1,
  'ex8.4': ex_8_4,
  'example8.4': example_8_4,
  '8.7': fig_8_7,
  '8.8': fig_8_8,
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
