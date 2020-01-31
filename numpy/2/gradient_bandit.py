from bandit import Bandit
import matplotlib.pyplot as plt
import numpy as np


def softmax(H):
  exp_val = np.exp(H)
  p = exp_val / np.sum(exp_val)
  return p


def gradient_update(H, pi, A, R, baseline=0, alpha=0.1):
  for a in range(len(H)):
    if a == A:
      H[a] += alpha * (R - baseline) * (1 - pi[a])
    else:
      H[a] -= alpha * (R - baseline) * pi[a]


def gradient_bandit(bandit, n_steps=1000, alpha=0.1, baseline=False,
                    random_walk=False):
  k = bandit.k
  H, per_max_act_log, R_mean, per_max_act = np.zeros(k), [], 0, 0
  max_action = bandit.max_action()
  for t in range(1, n_steps + 1):
    if random_walk:
      bandit.q += 0.01 * np.random.randn(bandit.k)
      max_action = bandit.max_action()
    pi = softmax(H)
    A = np.random.choice(len(H), p=pi)
    R = bandit.reward(A)
    R_mean += (R-R_mean) / t
    per_max_act += ((A == max_action) - per_max_act) / t
    per_max_act_log.append(per_max_act)
    baseline = R_mean if baseline else 0
    gradient_update(H, pi, A, R, baseline, alpha)
  return np.array(per_max_act_log)


def fig_2_5(n_bandits=2000, n_steps=1000, k=10, random_walk=False):
  d = {}
  for baseline in [False, True]:
    for alpha in [0.1, 0.4]:
      d[(baseline, alpha)] = np.zeros(n_steps)
      for n in range(n_bandits):
        print(n)
        bandit = Bandit(k, mean=4)
        d[(baseline, alpha)] += gradient_bandit(bandit,
                                                n_steps=n_steps,
                                                alpha=0.1, baseline=baseline,
                                                random_walk=random_walk)

  def label(baseline, alpha):
    return ("with" if baseline else "without") + f" baseline, alpha={alpha}"
  for key, avg_rew in d.items():
    plt.plot(avg_rew / n_bandits, label=label(key[0], key[1]))
  plt.legend()
  plt.show()


def main():
  fig_2_5(n_bandits=2000, random_walk=False)
  fig_2_5(n_bandits=2000, random_walk=True)


if __name__ == "__main__":
  main()
