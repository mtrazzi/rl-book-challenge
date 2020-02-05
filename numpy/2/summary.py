import argparse

from bandit import Bandit
from figures import a_simple_bandit_algorithm, constant_alpha, sample_average
from gradient_bandit import gradient_bandit
import matplotlib.pyplot as plt
import numpy as np

HYPERPARMS = {
  'epsilon-greedy': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
  'gradient bandit': [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2],
  'ucb': [1/16, 1/4, 1/2, 1, 2, 4],
  'optimistic greedy': [1/4, 1/2, 1, 2, 4],
}

COLORS = {
  'epsilon-greedy': 'r',
  'gradient bandit': '#1AD02B',
  'ucb': 'b',
  'optimistic greedy': 'k',
}


def apply_method(bandit, n_iterations, method_name, hyperparam, nonstat=False,
                 start_timestep=np.inf):
  if method_name in ['epsilon-greedy', 'ucb', 'optimistic greedy']:
    Q_1 = hyperparam if method_name == 'optimistic greedy' else 0
    eps = hyperparam if method_name == 'epsilon-greedy' else 0
    c = hyperparam if method_name == 'ucb' else None
    weight_fn = (constant_alpha(0.1) if (method_name == 'optimistic greedy' or
                 (method_name == 'epsilon-greedy' and nonstat))
                 else sample_average)
    method = 'ucb' if method_name == 'ucb' else 'epsilon-greedy'
    _, _, avg_rew, avg_rew_end = a_simple_bandit_algorithm(
                                              bandit,
                                              n_iterations=n_iterations,
                                              eps=eps, weight_fn=weight_fn,
                                              Q_1=Q_1, method=method, c=c,
                                              random_walk=nonstat,
                                              start_timestep=start_timestep)
  else:
    avg_rew, avg_rew_end = gradient_bandit(bandit, n_steps=n_iterations,
                                           alpha=hyperparam,
                                           baseline=True, percentage=False,
                                           start_timestep=start_timestep,
                                           random_walk=nonstat)
  return avg_rew_end if nonstat else avg_rew


def plot_current(n_steps, results, iteration_nb=0,
                 title='Figure 2.6', fn='fig2_6', y_label=''):
  _, ax = plt.subplots()
  ax.set_xscale('log', basex=2)
  x = [2 ** i for i in range(-7, 3)]
  x_name = ([f"1/{2**i}" for i in range(7, 0, -1)] +
            [str(2 ** i) for i in range(3)])
  plt.xticks(x, x_name)
  plt.ylabel(y_label)
  plt.title(title)
  for method, hyperparams in HYPERPARMS.items():
    to_plot = [results[(method, hyper)] / (iteration_nb + 1) for hyper in
               hyperparams]
    plt.plot(hyperparams, to_plot, color=COLORS[method], label=method)
  plt.legend()
  plt.savefig(f"figs/{fn}_{iteration_nb}.png")
  plt.close()


def param_study(n_bandits=2000, n_steps=1000, title='Figure 2.6',
                fn='fig2_6', nonstat=False, print_freq=10,
                start_timestep=np.inf):
  results = {(method, hyper): 0 for (method, hyperparams) in HYPERPARMS.items()
             for hyper in hyperparams}
  y_label = (f"Average Reward over last {n_steps-start_timestep} steps" if
             nonstat else f"Average Reward over first {n_steps} steps")
  for t in range(1, n_bandits + 1):
    print(f"{t}/{n_bandits}")
    bandit = Bandit()
    for method, hyperparams in HYPERPARMS.items():
      for hyper in hyperparams:
        results[(method, hyper)] += apply_method(bandit, n_steps, method, hyper,
                                                 nonstat, start_timestep)[-1]
        bandit.reset()  # need to reset q values after random walk
    if (t % print_freq == 0):
      plot_current(n_steps, results, t, title, fn, y_label)


def fig_2_6():
  param_study()


def ex_2_11():
  param_study(n_bandits=10, n_steps=int(2e5), title='Exercise 2.11',
              fn='ex2_11', nonstat=True, start_timestep=int(1e5), print_freq=1)


PLOT_FUNCTION = {
  '2.6': fig_2_6,
  'ex2.11': ex_2_11,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.', choices=['2.6', 'ex2.11'])
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure]()


if __name__ == "__main__":
  main()
