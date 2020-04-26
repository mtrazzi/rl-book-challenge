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
                    percentage=True, start_timestep=np.inf, random_walk=False):
  k, avg_r_end = bandit.k, 0
  H, per_max_act_log, avg_rew, R_mean, per_max_act = np.zeros(k), [], [], 0, 0
  max_action = bandit.max_action()
  for t in range(1, n_steps + 1):
    if random_walk:
      bandit.q += 0.01 * np.random.randn(k)
    pi = softmax(H)
    A = np.random.choice(len(H), p=pi)
    R = bandit.reward(A)
    per_max_act += ((A == max_action) - per_max_act) / t
    per_max_act_log.append(per_max_act)
    overline_R_t = R_mean if baseline else 0
    gradient_update(H, pi, A, R, overline_R_t, alpha)
    R_mean += (R-R_mean) / t  # baseline \overline{R_t} doesn't include R_t!
    avg_rew.append(R_mean)
    if t >= start_timestep:
      avg_r_end += (R - avg_r_end) / (t - start_timestep + 1)
  return (np.array(per_max_act_log) if percentage else np.array(avg_rew),
          [avg_r_end])


def fig_2_5(n_bandits=2000, n_steps=1000, k=10, alpha_list=[0.1, 0.4]):
  d = {}
  for baseline in [False, True]:
    for alpha in alpha_list:
      d[(baseline, alpha)] = np.zeros(n_steps)
  for n in range(n_bandits):
    print(n)
    bandit = Bandit(k, mean=4)
    for baseline in [False, True]:
      for alpha in alpha_list:
        result_arr, _ = gradient_bandit(bandit, n_steps=n_steps,
                                                alpha=alpha, baseline=baseline)
        d[(baseline, alpha)] += result_arr

  def label(baseline, alpha):
    return ("with" if baseline else "without") + f" baseline, alpha={alpha}"
  for key, avg_rew in d.items():
    plt.plot((avg_rew / n_bandits) * 100, label=label(key[0], key[1]))
  axes = plt.gca()
  axes.set_ylim([0, 100])
  plt.xlabel("Steps")
  plt.ylabel("Optimal Action %")
  plt.title("Figure 2.5")
  plt.legend()
  plt.show()


def main():
  fig_2_5()


if __name__ == "__main__":
  main()
