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


def gradient_bandit(bandit, n_steps=1000, alpha=0.1, baseline=False):
  k = bandit.k
  H, R_mean = np.zeros(k), 0
  avg_rew = []
  for t in range(1, n_steps + 1):
    bandit.q += 0.01 * np.random.randn(bandit.k)
    pi = softmax(H)
    A = np.random.choice(len(H), p=pi)
    R = bandit.reward(A)
    R_mean += (R-R_mean) / t
    avg_rew.append(R_mean)
    baseline = R_mean if baseline else 0
    gradient_update(H, pi, A, R, baseline, alpha)
  return np.array(avg_rew)

def main():
  k = 10
  n_steps = 1000
  n_bandits = 10
  d = {}
  for baseline in [False, True]:
    for alpha in [0.1, 0.4]:
      d[(baseline, alpha)] = np.zeros(n_steps)
      for _ in range(n_bandits):
        bandit = Bandit(k, mean=4)
        d[(baseline, alpha)] += gradient_bandit(bandit,
                                                n_steps=n_steps,
                                                alpha=0.1, baseline=baseline)
  def label(baseline, alpha):
    return ("with" if baseline else "without") + f" baseline, alpha={alpha}"
  for key, avg_rew in d.items():
    plt.plot(avg_rew / n_bandits, label=label(key[0], key[1]))
  plt.legend()
  plt.show()


if __name__ == "__main__":
  main()
