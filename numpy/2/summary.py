from bandit import Bandit
import matplotlib.pyplot as plt
import numpy as np

from gradient_bandit import gradient_bandit
from figures import a_simple_bandit_algorithm, constant_alpha, sample_average


HYPERPARMS = {
  'epsilon-greedy': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
  'gradient bandit': [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2],
  'ucb': [1/16, 1/4, 1/2, 1, 2, 4],
  'optimistic greedy': [1/4, 1/2, 1, 2, 4],
}

COLORS = {
  'epsilon-greedy': 'r',
  'gradient bandit': 'g',
  'ucb': 'b',
  'optimistic greedy': 'k',
}


def apply_method(bandit, n_iterations, method_name, hyperparam):
  if method_name in ['eps-greedy', 'ucb', 'optimistic greedy']:
    Q_1 = hyperparam if method_name == 'optimistic greedy' else 0
    eps = hyperparam if method_name == 'epsilon-greedy' else 0
    c = hyperparam if method_name == 'ucb' else None
    weight_fn = (constant_alpha if method_name == 'optimstic greedy' else
                 sample_average)
    method = 'ucb' if method_name == 'ucb' else 'epsilon-greedy'
    _, _, avg_rew = a_simple_bandit_algorithm(bandit, n_iterations=n_iterations,
                                              eps=eps, weight_fn=weight_fn,
                                              Q_1=Q_1, method=method, c=c)
  else:
    avg_rew = gradient_bandit(bandit, n_steps=n_iterations, alpha=hyperparam,
                              baseline=True, percentage=False)
  return avg_rew


def fig_2_6(n_bandits=2000, n_steps=1000):
  # axes = plt.gca()
  _, ax = plt.subplots()
  # axes.set_ylim([1, 1.5])
  ax.set_xscale('log', basex=2)
  plt.title('Figure 2.6')
  d = {(method, hyper): 0 for (method, hyperparams) in HYPERPARMS.items()
       for hyper in hyperparams}
  for t in range(n_bandits):
    print(f"{t}/{n_bandits}")
    bandit = Bandit()
    for method, hyperparams in HYPERPARMS.items():
      for hyper in hyperparams:
        d[(method, hyper)] += apply_method(bandit, n_steps, method, hyper)[-1]
  for method, hyperparams in HYPERPARMS.items():
    to_plot = [d[(method, hyper)] / n_bandits for hyper in hyperparams]
    plt.plot(hyperparams, to_plot, color=COLORS[method], label=method)
  plt.legend()

  plt.ylabel(f"Average Reward over first {n_steps} steps")
  plt.savefig("fig2_6.png")
  plt.show()


def main():
  fig_2_6(10)


if __name__ == "__main__":
  main()
