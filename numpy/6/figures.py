import argparse
from td import OneStepTD
from off_pol_td import OffPolicyTD
from driving import DrivingEnv, TRAVEL_TIME
from sarsa import Sarsa
from windy_gridworld import WindyGridworld
import numpy as np
from randomwalk import RandomWalk, NotSoRandomWalk, LEFT, RIGHT
import matplotlib.pyplot as plt

N_EP_EX_6_2 = 100
N_RUNS_EX_6_2 = 100
TRUE_VALUES_EX_6_2 = [1/6, 2/6, 3/6, 4/6, 5/6]
TD_STEPS_6_2 = [0.05, 0.1, 0.15]
MC_STEPS_6_2 = [0.01, 0.02, 0.03, 0.04] 
TD_STEPS_6_4 = [0.025, 0.05, 0.1, 0.15, 0.2]
MC_STEPS_6_4 = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05] 
INIT_VAL_6_2 = 1/2
LEFT_GRAPH_STEP_SIZE = 0.1
DEFAULT_FONT = {'fontsize': 14}
SMALL_FONT = {'fontsize': 10}
UNDISCOUNTED = 1
BATCH_ALPHA = {'td': 0.002, 'mc': 0.001}
NOT_SO_RW_ALPHA = 0.001
EX_6_5_STEP_SIZE = 0.5
EX_6_5_EPS = 0.1
EX_6_5_XTICKS = [k * 1000 for k in range(9)]
EX_6_5_YTICKS = [0, 50, 100, 150, 170]
EX_6_9_XTICKS = [k * 2000 for k in range(20)]
EX_6_9_YTICKS = [0, 50, 100, 150, 170, 200, 500, 1000]
EX_6_10_N_SEEDS = 10
 
def print_driving_home(states, V_old, V_new, fig, fig_id, ax_title):
  ax = fig.add_subplot(fig_id)
  ax.set_title(ax_title)
  def pred(V):
    return [V[idx] + sum(TRAVEL_TIME[:idx]) for idx in range(len(V))]
  ax.set_xticklabels(states, fontdict={'fontsize': 8})
  ax.set_xlabel('Situation')
  ax.set_ylabel('Predicted total travel time')
  plt.plot(pred(V_old), color='#000000', label='actual outcome')
  plt.plot(pred(V_new), color='blue', label='after update')

def fig_6_1():
  fig = plt.figure()
  fig.suptitle('Figure 6.1')
  env = DrivingEnv()
  pi = {(a, s): 1.0 for s in env.states for a in env.moves} 
  V_0 = [30, 35, 15, 10, 3, 0]
  V_init = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
 
  # TD(0)
  alg = OneStepTD(env, V_init=V_init, step_size=1, gamma=1)
  alg.tabular_td_0(pi)
  V_TD = alg.get_value_list()
  print_driving_home(env.states, V_0, V_TD, fig, '121', 'one step TD')
 
  # constant step size mc
  alg.reset()
  alg.constant_step_size_mc(pi)
  V_MC = alg.get_value_list()
  print_driving_home(env.states, V_0, V_MC, fig, '122', 'constant step size mc')

  plt.legend()
  plt.savefig('fig6.1.png')
  plt.show()

def print_random_walk(ax, state_labels, td_vals):
  ax.set_xticklabels(state_labels, fontdict=DEFAULT_FONT)
  x_ticks = np.arange(len(state_labels))
  ax.set_xticks(x_ticks)
  ax.set_xlabel('state', fontdict=DEFAULT_FONT)
  ax.set_ylabel('estimated value', fontdict=DEFAULT_FONT) 
  plt.plot(x_ticks, TRUE_VALUES_EX_6_2, label='true values')
  for key,td_val in td_vals.items():
    plt.plot(x_ticks, td_val[:-1], label=str(key) + ' episodes')

def init_random_walk(init_value, step_size=None):
  env = RandomWalk()
  pi = {(a, s): 1.0 for s in env.states for a in env.moves} 
  V_0 = [init_value for s in env.states[:-1]] + [0]  # V = 0 for absorbing state
  V_init = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
  alg = OneStepTD(env, V_init=V_init, step_size=step_size, gamma=UNDISCOUNTED)
  return alg, pi

def left_graph(fig, fig_id, init_value):
  alg, pi = init_random_walk(init_value, step_size=LEFT_GRAPH_STEP_SIZE)
  tot_ep = 0
  td_vals = {}
  ax = fig.add_subplot('121')
  for n_episodes in [0, 1, 10, 100]:
    alg.tabular_td_0(pi, n_episodes - tot_ep)
    td_vals[n_episodes] = alg.get_value_list()
  print_random_walk(ax, ["A", "B", "C", "D", "E"], td_vals)
  plt.legend()

def right_graph(fig, fig_id, init_value, td_step_sizes, mc_step_sizes, font=DEFAULT_FONT, remove_x_label=False, batch=False): 
  ax = fig.add_subplot(fig_id)
  ax.set_title(f'V_init = {init_value}', fontdict=font)
  alg, pi = init_random_walk(init_value)
  runs_dict = {alpha: np.zeros(N_EP_EX_6_2) for alpha in td_step_sizes + mc_step_sizes} 
  td_0 = alg.tabular_td_0 if not batch else alg.td_0_batch
  mc = alg.constant_step_size_mc if not batch else alg.constant_step_size_mc_batch
  to_compare_list = [(td_step_sizes, td_0), (mc_step_sizes, mc)]
  for (step_size_list, algorithm) in to_compare_list:
    for step_size in step_size_list:
      alg.step_size = step_size
      print(f"running step size {step_size}")
      for seed in range(N_RUNS_EX_6_2): 
        alg.reset()
        alg.env.seed(seed)
        err_l = []
        for nb_ep in range(N_EP_EX_6_2):
          algorithm(pi, 1)
          v_arr = np.array(alg.get_value_list()[:-1])
          err_l.append(np.linalg.norm(v_arr-TRUE_VALUES_EX_6_2))
        runs_dict[step_size] += np.array(err_l)

  for key in runs_dict.keys():
    runs_dict[key] /= N_RUNS_EX_6_2
  
  if not remove_x_label:
    ax.set_xlabel('walks / episodes', fontdict=font)
  ax.set_ylabel('empirical rms error averaged over states', fontdict=font) 
  for key,err_run in runs_dict.items():
    (color, alg_name) = ('b','td') if key in td_step_sizes else ('r', 'mc')
    linewidth = max(int(100 * key) / 10 if key in td_step_sizes else int(200 * key) / 10, 10 / (len(runs_dict) * 10))
    linestyle = 'dashed' if key in [0.02, 0.03] else None
    plt.plot(err_run, color=color, label=alg_name + ' (a=' + str(key) + ')', linewidth=linewidth, linestyle=linestyle)
   
  plt.legend()

def example_6_2():
  fig = plt.figure()
  fig.suptitle('Example 6.2', fontdict=DEFAULT_FONT)
  left_graph(fig, fig_id='121', init_value=INIT_VAL_6_2)
  right_graph(fig, '122', INIT_VAL_6_2, TD_STEPS_6_2, MC_STEPS_6_2)
  plt.savefig('example6.2.png')
  plt.show()

def ex_6_4():
  fig = plt.figure()
  fig.suptitle('Exercise 6.4', fontdict=DEFAULT_FONT)
  right_graph(fig, '111', INIT_VAL_6_2, TD_STEPS_6_4, MC_STEPS_6_4, SMALL_FONT)
  plt.savefig('ex6.4.png')
  plt.show()

def ex_6_5():
  fig = plt.figure()
  fig.suptitle('Exercise 6.5', fontdict=SMALL_FONT)
  for (idx, init_val) in enumerate([0, 0.25, 0.75, 1]):
    right_graph(fig, '22' + str(idx + 1), init_val, TD_STEPS_6_2, MC_STEPS_6_2, SMALL_FONT, idx < 2)
  plt.savefig('ex6.5.png')
  plt.show()

def fig_6_2():
  fig = plt.figure()
  fig.suptitle('Figure 6.2', fontdict=SMALL_FONT)
  right_graph(fig, '111', INIT_VAL_6_2, [BATCH_ALPHA['td']], [BATCH_ALPHA['mc']], batch=True, font=SMALL_FONT)
  plt.savefig('fig6.2.png')
  plt.show()

def ex_6_7():
  env = NotSoRandomWalk()
  env.seed(0)
  V_0 = [1/2 for s in env.states[:-1]] + [0]
  V_init = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
  b = {(a, s): 1/2 for s in env.states for a in env.moves} 
  pi = {(a, s): float(a == RIGHT) for s in env.states for a in env.moves}
  alg = OffPolicyTD(env, V_init, NOT_SO_RW_ALPHA, pi, b, UNDISCOUNTED)
  alg.step_size = 0.01
  alg.find_value_function(N_EP_EX_6_2 * 100)
  print(alg.get_value_list())

def init_windy_gridworld_fig(title, xticks=None, yticks=None):
  fig, ax = plt.subplots() 
  fig.suptitle(title)
  ax.set_xlabel('Time steps')
  ax.set_ylabel('Episodes')
  if xticks is not None:
    ax.set_xticks(xticks)
  if yticks is not None:
    ax.set_yticks(yticks)
  return ax

def plot_sarsa(ax, n_ep, label=None, diags=False, stay=False, stoch=False, seed=0):
  env = WindyGridworld(diags, stay, stoch)
  alg = Sarsa(env, step_size=EX_6_5_STEP_SIZE, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) 
  alg.seed(seed)
  kwargs = {"label": label} if label else {}
  plt.plot(alg.on_policy_td_control(n_ep), **kwargs)

def example_6_5():
  ax = init_windy_gridworld_fig('Example 6.5', EX_6_5_XTICKS, EX_6_5_YTICKS)
  plot_sarsa(ax, max(EX_6_5_YTICKS))
  plt.savefig('example6.5.png')
  plt.show()

def ex_6_9():
  ax = init_windy_gridworld_fig('Exercise 6.9', EX_6_9_XTICKS, EX_6_9_YTICKS)
  n_ep_urld, n_ep = EX_6_9_YTICKS[-2:]
  plot_sarsa(ax, n_ep_urld, label='up right down left')
  plot_sarsa(ax, n_ep, label='with diags', diags=True)
  plot_sarsa(ax, n_ep, label='with diags and stay', diags=True, stay=True)
  plt.legend()
  plt.savefig('ex6.9.png')
  plt.show()

def ex_6_10():
  ax = init_windy_gridworld_fig(f'Exercise 6.10 ({EX_6_10_N_SEEDS} seeds)')
  n_ep = max(EX_6_9_YTICKS)
  for seed in range(EX_6_10_N_SEEDS):
    plot_sarsa(ax, n_ep, diags=True, stay=True, stoch=True, seed=seed)
  plt.savefig('ex6.10.png')
  plt.show()

PLOT_FUNCTION = {
  '6.1': fig_6_1,
  'example6.2': example_6_2,
  'ex6.4': ex_6_4,
  'ex6.5': ex_6_5,
  '6.2': fig_6_2,
  'ex6.7': ex_6_7,
  'example6.5': example_6_5,
  'ex6.9': ex_6_9,
  'ex6.10': ex_6_10, 
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
