import argparse
from td import OneStepTD
from driving import DrivingEnv, TRAVEL_TIME
import numpy as np
from randomwalk import RandomWalk
import matplotlib.pyplot as plt

N_EP_EX_6_2 = 100
N_RUNS_EX_6_2 = 100
TRUE_VALUES_EX_6_2 = [1/6, 2/6, 3/6, 4/6, 5/6]
DEFAULT_FONT = {'fontsize': 14}

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

def example_6_2():
  fig = plt.figure()
  fig.suptitle('Example 6.2', fontdict=DEFAULT_FONT)
  env = RandomWalk()
  pi = {(a, s): 1.0 for s in env.states for a in env.moves} 
  V_0 = [1/2 for s in env.states[:-1]] + [0]  # V = 0 for absorbing state
  V_init = {s: V_0[idx] for (idx, s) in enumerate(env.states)}
  
  alg = OneStepTD(env, V_init=V_init, step_size=0.1, gamma=1)
  tot_ep = 0
  td_vals = {}
  ax = fig.add_subplot('121')
  for n_episodes in [0, 1, 10, 100]:
    alg.tabular_td_0(pi, n_episodes - tot_ep)
    td_vals[n_episodes] = alg.get_value_list()
  print_random_walk(ax, ["A", "B", "C", "D", "E"], td_vals)
  plt.legend()
  
  ax = fig.add_subplot('122')
  td_step_sizes = [0.05, 0.1, 0.15]
  td_vals = {alpha: [] for alpha in td_step_sizes}
  mc_step_sizes = [0.01, 0.02, 0.03, 0.04]
  mc_vals = {alpha: [] for alpha in mc_step_sizes}
  runs_dict = {alpha: np.zeros(N_EP_EX_6_2) for alpha in td_step_sizes + mc_step_sizes} 
  to_compare_list = [(td_step_sizes, alg.tabular_td_0, td_vals), (mc_step_sizes, alg.constant_step_size_mc, mc_vals)]
  for (step_size_list, algorithm, store_dict) in to_compare_list:
    for step_size in step_size_list:
      alg.step_size = step_size
      print(f"running step size {step_size}")
      for seed in range(N_RUNS_EX_6_2): 
        alg.reset()
        alg.env.seed(seed)
        err_l = []
        for _ in range(N_EP_EX_6_2):
          algorithm(pi, 1)
          v_arr = np.array(alg.get_value_list()[:-1])
          err_l.append(np.linalg.norm(v_arr-TRUE_VALUES_EX_6_2))
        runs_dict[step_size] += np.array(err_l)

  for key in runs_dict.keys():
    runs_dict[key] /= N_RUNS_EX_6_2
  
  ax.set_xlabel('walks / episodes', fontdict=DEFAULT_FONT)
  ax.set_ylabel('empirical rms error averaged over states', fontdict=DEFAULT_FONT) 
  for key,err_run in runs_dict.items():
    (color, alg_name) = ('b','td') if key in td_step_sizes else ('r', 'mc')
    linewidth = int(100 * key) / 10 if key in td_step_sizes else int(200 * key) / 10
    linestyle = 'dashed' if key in [0.02, 0.03] else None
    plt.plot(err_run, color=color, label=alg_name + ' (a=' + str(key) + ')', linewidth=linewidth, linestyle=linestyle)
   
  plt.legend()
  plt.savefig('example6.2.png')
  plt.show()


PLOT_FUNCTION = {
  '6.1': fig_6_1,
  'example6.2': example_6_2,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  args = parser.parse_args()

  if args.figure in ['6.1', 'example6.2']:
    PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
