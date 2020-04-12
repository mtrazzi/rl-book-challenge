import argparse
from td import OneStepTD
from driving import DrivingEnv, TRAVEL_TIME
from randomwalk import RandomWalk
import matplotlib.pyplot as plt

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
  ax.set_xticklabels(state_labels, fontdict={'fontsize': 8})
  ax.set_xlabel('state')
  ax.set_ylabel('estimated value') 
  true_values = [(x + 1)/6 for x in range(len(state_labels))]
  plt.plot(true_values, label='true values')
  for key,td_val in td_vals.items():
    plt.plot(td_val, label=key)

def example_6_2():
  fig = plt.figure()
  fig.suptitle('Example 6.2')
  env = RandomWalk()
  pi = {(a, s): 1.0 for s in env.states for a in env.moves} 
  V_0 = [1/2 for s in env.states]
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
  plt.savefig('example6.2.png')
  plt.show()
 
  ## constant step size mc
  #alg.reset()
  #alg.constant_step_size_mc(pi)
  #V_MC = alg.get_value_list()
  #print_driving_home(env.states, V_0, V_MC, fig, '122', 'constant step size mc')



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
