import argparse
from td import OneStepTD
from off_pol_td import OffPolicyTD
from driving import DrivingEnv, TRAVEL_TIME
from sarsa import Sarsa
from windy_gridworld import WindyGridworld
import numpy as np
from randomwalk import RandomWalk, NotSoRandomWalk, LEFT, RIGHT
from cliff import TheCliff
import matplotlib.pyplot as plt
from qlearning import QLearning
from double_qlearning import DoubleQLearning
from expected_sarsa import ExpectedSarsa
from max_bias_mdp import MaxBiasMDP, S_A, LEFT
from double_expected_sarsa import DoubleExpectedSarsa
from car_rental_afterstate import CarRentalAfterstateEnv
from td_afterstate import TDAfterstate
from policy_iteration_afterstate import DynamicProgrammingAfterstate
import seaborn as sns

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
EX_6_6_N_EPS = 500
EX_6_6_YTICKS = [-100, -75, -50, -25]
EX_6_6_N_SEEDS = 10
EX_6_6_N_AVG = 50
FIG_6_3_N_INT_RUNS = 250
FIG_6_3_N_INT_EPS = 100
FIG_6_3_N_ASY_RUNS = 5
FIG_6_3_N_ASY_EPS = 1000
FIG_6_5_ALPHA = 0.1
FIG_6_5_N_RUNS = 100
FIG_6_5_N_EPS = 300
EX_6_13_N_EPS = 300
EX_6_13_N_RUNS = 10000
EX_6_14_SIZE = 4
EX_6_14_ALPHA = 0.01
EX_6_14_N_EPS = 1000
EX_6_14_GAMMA = 0.9
 
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

def smooth_rewards(arr, to_avg=5):
  nb_rew = len(arr)
  new_arr = np.zeros(nb_rew - to_avg + 1) 
  for i in range(nb_rew - to_avg + 1):
    new_arr[i] = np.mean([arr[i + j] for j in range(to_avg)])
  return new_arr

def example_6_6():
  fig, ax = plt.subplots() 
  fig.suptitle(f'Example 6.6 (Averaged over {EX_6_6_N_SEEDS} seeds)')
  ax.set_xlabel('Episodes')
  ax.set_ylabel(f'(Average of last {EX_6_6_N_AVG}) sum of rewards during episodes')
  ax.set_yticks(EX_6_6_YTICKS)
  ax.set_ylim(bottom=min(EX_6_6_YTICKS))
  n_ep = EX_6_6_N_EPS
  env = TheCliff()
  qlearning_alg = QLearning(env, step_size=EX_6_5_STEP_SIZE, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) 
  sarsa_alg = Sarsa(env, step_size=EX_6_5_STEP_SIZE, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) 
  qlearning_rew = np.zeros(n_ep)
  sarsa_rew = np.zeros(n_ep)
  for seed in range(EX_6_6_N_SEEDS):
    print(f"seed={seed}")
    qlearning_alg.seed(seed)
    qlearning_rew += qlearning_alg.q_learning(n_ep)
    sarsa_alg.seed(seed)
    sarsa_rew += sarsa_alg.on_policy_td_control(n_ep, rews=True)
  plt.plot(smooth_rewards(qlearning_rew / EX_6_6_N_SEEDS, EX_6_6_N_AVG), color='r', label='Q learning')
  plt.plot(smooth_rewards(sarsa_rew / EX_6_6_N_SEEDS, EX_6_6_N_AVG), color='b', label='Sarsa')
  plt.legend()
  plt.savefig('example6.6.png')
  plt.show()

def fig_6_3(): 
  fig, ax = plt.subplots() 
  fig.suptitle(f'Figure 6.3')
  step_sizes = np.linspace(0.1, 1, 19)
  ax.set_xlabel(f'Step Sizes')
  ax.set_xticks(step_sizes)
  ax.set_yticks([0, -40, -80, -120])
  ax.set_ylim(bottom=-160, top=0)
  ax.set_ylabel('Sum of rewards per episodes')
  env = TheCliff() 
  exp_sar_alg, sar_alg, qlear_alg = [name_alg(env, step_size=None, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) for name_alg in [ExpectedSarsa, Sarsa, QLearning]]
  exp_sar_opt, sar_opt, qlear_opt = exp_sar_alg.expected_sarsa, lambda n_ep: sar_alg.on_policy_td_control(n_ep, rews=True), qlear_alg.q_learning
  for (alg, opt, alg_name, color, marker) in [(exp_sar_alg, exp_sar_opt, 'Expected Sarsa', 'r', 'x'), (sar_alg, sar_opt, 'Sarsa', 'b', 'v'), (qlear_alg, qlear_opt, 'Q-learning', 'k', 's')]:
    print(f"\n\n\n@@@@@@@@ {alg_name} @@@@@@@@\n@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\n")
    for (n_ep, n_runs, run_type_name) in [(FIG_6_3_N_INT_EPS, FIG_6_3_N_INT_RUNS, 'Interim'), (FIG_6_3_N_ASY_EPS, FIG_6_3_N_ASY_RUNS, 'Asymptotic')]:
      print(f"\n######## {run_type_name} ########\n")
      rew_l = []
      for step_size in step_sizes: 
        print(f"alpha={step_size}")
        alg.step_size = step_size
        rew_sum = 0
        for seed in range(n_runs):
          print(f"run #{seed}")
          alg.seed(seed)
          alg.reset()
          rew_sum += np.mean(opt(n_ep))
        rew_l.append(rew_sum / n_runs)
      label = f"{alg_name} ({run_type_name})"
      plt.plot(step_sizes, rew_l, label=label, color=color, marker=marker, linestyle='-' if run_type_name == 'Asymptotic' else '--')
  plt.legend()
  plt.savefig('fig6.3.png')
  plt.show()


def plot_max_bias(title, filename, todo_list, n_runs, n_eps):
  fig, ax = plt.subplots() 
  fig.suptitle(title)
  ax.set_xlabel(f'Episodes')
  xticks = [1, 100, 200, 300]
  ax.set_xlim([min(xticks), max(xticks)])
  ax.set_yticks([0, 5, 25, 50, 75, 100])
  ax.set_ylim([0, 100])
  ax.set_ylabel('% left actions from A')
  for (alg, opt, color, label) in todo_list:
    perc_left = np.zeros(n_eps)
    for seed in range(n_runs):
      print(seed)
      alg.seed(seed)
      alg.reset()
      perc_left += opt(n_eps)
    plt.plot(perc_left / n_runs, label=label, color=color)
  plt.plot(np.zeros(n_eps) + 5, color='k', linestyle='--', label='optimal')
  plt.legend()
  plt.savefig(filename)
  plt.show()

def fig_6_5():
  env = MaxBiasMDP()
  qlear_alg, dqlear_alg = [name_alg(env, step_size=FIG_6_5_ALPHA, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) for name_alg in [QLearning, DoubleQLearning]]
  qlear_opt = lambda n_ep: qlear_alg.q_learning_log_actions(n_ep, S_A, LEFT)
  dqlear_opt = lambda n_ep: dqlear_alg.double_q_learning_log_actions(n_ep, S_A, LEFT)
  todo = [(qlear_alg, qlear_opt, 'r', 'Q-learning'), (dqlear_alg, dqlear_opt, 'g', 'Double Q-learning')]
  plot_max_bias('Figure 6.5', 'fig6.5.png', todo, FIG_6_5_N_RUNS, FIG_6_5_N_EPS)

def ex_6_13():
  env = MaxBiasMDP()
  esarsa_alg, desarsa_alg = [name_alg(env, step_size=FIG_6_5_ALPHA, gamma=UNDISCOUNTED, eps=EX_6_5_EPS) for name_alg in [ExpectedSarsa, DoubleExpectedSarsa]]
  esarsa_opt = lambda n_ep: esarsa_alg.expected_sarsa_log_actions(n_ep, S_A, LEFT)
  desarsa_opt = lambda n_ep: desarsa_alg.double_expected_sarsa_log_actions(n_ep, S_A, LEFT)
  todo = [(desarsa_alg, desarsa_opt, 'g', 'Double Expected Sarsa'), (esarsa_alg, esarsa_opt, 'r', 'Expected Sarsa')]
  plot_max_bias(f'Exercise 6.13 ({EX_6_13_N_RUNS} runs)', 'ex6.13.png', todo, EX_6_13_N_RUNS, EX_6_13_N_EPS)

def print_car_rental_value_function(size, V): 
  to_print = np.zeros((size, size))
  idxs = list(range(size))
  for x in idxs:
    for y in idxs:
      to_print[x][y] = V[(x, y)]
  to_print_term = [[to_print[size - x - 1][y] for y in idxs] for x in idxs]
  print(f"#####\n\nV mean = {np.mean(to_print_term)}\n\n######")
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #plt.title('Exercise 6.14 (value function)')
  #(X, Y), Z = np.meshgrid(idxs, idxs), np.array(to_print).T
  #ax.set_xlabel('# of cars at second location', fontsize=10)
  #ax.set_ylabel('# of cars at first location', fontsize=10)
  #ax.set_xticks([idxs[0], idxs[-1]])
  #ax.set_yticks([idxs[0], idxs[-1]])
  #ax.set_zticks([np.min(Z), np.max(Z)])
  #ax.plot_surface(X, Y, Z)
  #plt.show()
  return np.mean(to_print_term)

def print_policy_car_rental(size, pi):
  fig, ax = plt.subplots()
  X = Y = list(range(size))
  Z = [[pi[(x, y)] for y in Y] for x in X]
  transposed_Z = [[Z[size - x - 1][y] for y in Y] for x in X]
  sns.heatmap(transposed_Z)
  print(*transposed_Z, sep='\n')
  pol_range = list(range(np.min(transposed_Z), np.max(transposed_Z) + 1))
  #CS = ax.contour(X, Y, Z, colors='k', levels=pol_range)
  #ax.clabel(CS, inline=1, fontsize=10)
  ax.set_title('Exercise 6.14 (policy)')
  #plt.show()

def ex_6_14(size=None, ep_per_eval=None, alpha=None, max_ep=None):
  size = EX_6_14_SIZE if size is None else size
  env = CarRentalAfterstateEnv(size - 1)
  env.seed(0)
  #pi = {(a, s): (a == 0) for s in env.states for a in env.moves_d[s]}
  pi = {s: 0 for s in env.states}
  step_size_l = [0.003, 0.004, 0.005]
  log_V_mean = {step_size: [] for step_size in step_size_l}
  for step_size in step_size_l:
    tot_ep = 0
    alg = TDAfterstate(env, None, step_size=step_size, gamma=EX_6_14_GAMMA, pi_init=pi)
    stable = False
    while len(log_V_mean[step_size]) < 10:
      print(f"tot_ep = {tot_ep}")
      V, pi, stable = alg.policy_iteration(ep_per_eval=ep_per_eval, batch=True, max_ep=max_ep)
      tot_ep += ((ep_per_eval) * (ep_per_eval + 1)) // 2
      mean = print_car_rental_value_function(size, V)
      log_V_mean[step_size].append(mean)
      plt.savefig(f'ex6.14_val_{str(ep_per_eval)}_{str(alpha)[2:]}_{str(tot_ep)}ep.png')
      plt.close()
  for step_size in step_size_l:
    plt.plot(log_V_mean[step_size], label=f'alpha={step_size}')
  plt.legend()
  plt.savefig('learning_rates.png')
  plt.show()
  #print_policy_car_rental(size, pi)
  #plt.savefig('ex6.14_pol.png')

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
  'example6.6': example_6_6,
  '6.3': fig_6_3,
  '6.5': fig_6_5,
  'ex6.13': ex_6_13,
  'ex6.14': ex_6_14,
}

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=PLOT_FUNCTION.keys())
  parser.add_argument('-s', '--size', type=int, default=None,
                      help='Size of the environment (size * size states).')
  parser.add_argument('-e', '--ep', type=int, default=None)
  parser.add_argument('-a', '--alpha', type=float, default=None)
  parser.add_argument('-m', '--max_ep', type=int, default=None)
  args = parser.parse_args()

  if args.figure == 'ex6.14':
    PLOT_FUNCTION[args.figure](args.size, args.ep, args.alpha, args.max_ep)
  else:
    PLOT_FUNCTION[args.figure]()

if __name__ == "__main__":
  main()
