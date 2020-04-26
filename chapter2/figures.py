import argparse

from bandit import Bandit
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('TkAgg')

K = 10


def average_reward(Q, N):
  return np.dot(Q, N) / sum(N)


def constant_alpha(alpha=0.1):
  return lambda x: alpha


def sample_average(N):
  return (1/N)


def action_selection(Q, eps=0.1, method='epsilon-greedy', c=2, t=1, N=None):
  if method == 'epsilon-greedy':
    if np.random.random() < eps:
      return np.random.choice(Q.shape[0])
    else:
      return np.random.choice(np.flatnonzero(Q == Q.max()))
  elif method == 'ucb':
    ucb = np.zeros_like(Q)
    for a in range(Q.shape[0]):
      ucb[a] = Q[a] + c * np.sqrt(np.log(t) / N[a]) if N[a] > 0 else np.inf
    return np.random.choice(np.flatnonzero(ucb == ucb.max()))


def a_simple_bandit_algorithm(bandit, n_iterations=1000, eps=0.1,
                              weight_fn=sample_average, random_walk=False,
                              Q_1=0, method='epsilon-greedy', c=2,
                              start_timestep=np.inf):
  """Returns the estimated Q-Values of the bandit problem."""
  k = bandit.k
  Q, N, R_log = np.zeros(k) + Q_1, np.zeros(k), np.zeros(k)
  avg_rew, per_list = [], []
  avg_r, per_max_act, avg_r_end = 0, 0, 0
  for t in range(1, n_iterations + 1):
    A = action_selection(Q, eps, method=method, c=c, t=t, N=N)
    R = bandit.reward(A)
    N[A] += 1
    Q[A] += (R - Q[A]) * weight_fn(N[A])
    R_log[A] += (R-R_log[A]) * (1 / N[A])
    avg_r += (R - avg_r) / t
    if t >= start_timestep:
      avg_r_end += (R - avg_r_end) / (t - start_timestep + 1)
    per_max_act += ((A == bandit.max_action()) - per_max_act) / t
    per_list.append(per_max_act)
    avg_rew.append(avg_r)
    if random_walk:
      bandit.q += 0.01 * np.random.randn(k)
  return Q, np.array(per_list), np.array(avg_rew), [avg_r_end]


def plot_average(arr, eps_list, n_bandits, y_lim, show=True, extra_label='',
                 title=None, percentage=False):
  for i, eps in enumerate(eps_list):
    plt.plot((arr[i][1:] / n_bandits) * (100 if percentage else 1),
             label=f"epsilon={eps} {extra_label}")
  axes = plt.gca()
  axes.set_ylim(y_lim)
  plt.xlabel("Steps")
  plt.ylabel("Optimal Action %" if percentage else "Average Reward")
  plt.legend()
  if title is not None:
    plt.title(title)
  if show:
    plt.show()


def plot_figures(k, n_bandits, n_steps, eps_list, weight_fn=sample_average,
                 random_walk=False, y_bounds=[0, 1.5], Q_1=0, show=True,
                 method='epsilon-greedy', extra_label='', title=None,
                 percentage=False):
  avg_rew_per_eps = [np.zeros(n_steps) for _ in range(len(eps_list))]
  avg_rew_in_perc = [np.zeros(n_steps) for _ in range(len(eps_list))]
  for i in range(n_bandits):
    print(i)
    bandit_pb = Bandit(k)
    for i, eps in enumerate(eps_list):
      _, per, avg_rew, _ = a_simple_bandit_algorithm(bandit_pb,
                                                     n_iterations=n_steps,
                                                     eps=eps,
                                                     weight_fn=weight_fn,
                                                     random_walk=random_walk,
                                                     Q_1=Q_1,
                                                     method=method)
      avg_rew_per_eps[i] += avg_rew
      avg_rew_in_perc[i] += per

  to_plot = avg_rew_in_perc if percentage else avg_rew_per_eps
  bounds = [0, 100] if percentage else y_bounds
  plot_average(to_plot, eps_list, n_bandits, bounds, show, extra_label, title,
               percentage)


def fig_2_2(n_bandits=2000, n_steps=1000, eps_list=[0, 0.1, 0.01]):
  # reproducing figure 2.2
  plot_figures(K, n_bandits, n_steps, eps_list, title='Figure 2.2')
  plot_figures(K, n_bandits, n_steps, eps_list, title='Figure 2.2',
               percentage=True)


def ex_2_5(n_bandits=100, n_steps=10000, eps_list=[0.1]):
  # # exercise 2.5: difficulties of sample average on non-stationary problems
  plot_figures(K, n_bandits, n_steps, eps_list, sample_average, True, [0, 3],
               show=False, extra_label='sample average',
               title="Exercise 2.5: sample-average vs.constant step size")
  plot_figures(K, n_bandits, n_steps, eps_list, constant_alpha(alpha=0.1), True,
               [0, 3], extra_label='constant step size (alpha = 0.1)')


def fig_2_3(n_bandits=2000, n_steps=1000):
  # figure 2.3: optimistic greedy vs. realistic eps-greedy
  for Q_1, eps, show in [(5, 0, False), (0, 0.1, True)]:
    plot_figures(K, n_bandits, n_steps, [eps], constant_alpha(alpha=0.1),
                 Q_1=Q_1, show=show, title='Figure 2.3',
                 extra_label=f'Q_1={Q_1}', percentage=True)


def fig_2_4(n_bandits=2000, n_steps=1000, eps_list=[0.1]):
  # reproducing figure 2.4
  plot_figures(K, n_bandits, n_steps, eps_list, sample_average, False, [0, 1.5],
               show=False)
  plot_figures(K, n_bandits, n_steps, eps_list, sample_average, False, [0, 1.5],
               method='ucb', extra_label='ucb')


PLOT_FUNCTION = {
  '2.2': fig_2_2,
  'ex2.5': ex_2_5,
  '2.3': fig_2_3,
  '2.4': fig_2_4,
}


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('figure', type=str, default=None,
                      help='Figure to reproduce.',
                      choices=['2.2', 'ex2.5', '2.3', '2.4'])
  args = parser.parse_args()

  PLOT_FUNCTION[args.figure]()


if __name__ == "__main__":
  main()
