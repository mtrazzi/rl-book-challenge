from bandit import Bandit
import matplotlib.pyplot as plt
import numpy as np


def average_reward(Q, N):
  return np.dot(Q, N) / sum(N)


def a_simple_bandit_algorithm(bandit, n_iterations=1000, eps=0.1):
  """Returns the estimated Q-Values of the bandit problem."""
  Q, N = [np.zeros(bandit.k) for _ in range(2)]
  avg_rew = []
  for _ in range(n_iterations):
    if np.random.random() < eps:
      A = np.random.choice(bandit.k)
    else:
      A = np.random.choice(np.flatnonzero(Q == Q.max()))
    R = bandit.reward(A)
    N[A] += 1
    Q[A] += (R - Q[A]) / N[A]
    avg_rew.append(average_reward(Q, N))
  return Q, N, avg_rew


def plot_average(arr, eps_list, n_bandits, y_lim):
  for i, eps in enumerate(eps_list):
    plt.plot(arr[i][1:] / n_bandits, label=f"eps={eps}")
  axes = plt.gca()
  axes.set_ylim(y_lim)
  plt.legend()
  plt.show()


def main():
  # reproducing figure 2.2
  eps_list = [0, 0.1, 0.01]
  k = 10
  n_bandits = 20
  n_iter_per_bandit = 1000
  avg_rew_per_eps = [np.zeros(n_iter_per_bandit) for _ in range(len(eps_list))]
  avg_rew_in_perc = [np.zeros(n_iter_per_bandit) for _ in range(len(eps_list))]
  for i in range(n_bandits):
    print(i)
    bandit_pb = Bandit(k)
    for i, eps in enumerate(eps_list):
      _, _, avg_rew = a_simple_bandit_algorithm(bandit_pb,
                                                n_iterations=n_iter_per_bandit,
                                                eps=eps)
      avg_rew_per_eps[i] += avg_rew
      avg_rew_in_perc[i] += (avg_rew / bandit_pb.max_action)
  plot_average(avg_rew_per_eps, eps_list, n_bandits, [0, 1.5])
  plot_average(avg_rew_in_perc * 100, eps_list, n_bandits, [0, 1])


if __name__ == "__main__":
  main()
