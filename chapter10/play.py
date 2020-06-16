import argparse
import os
from mountain_car import MountainCar
import matplotlib.pyplot as plt

ENV_DICT = {
  'mountain_car': MountainCar(mnt=False),
}


def play(env):
  def refresh():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(env)
  while True:
    env.reset()
    done = False
    v = []
    while not done:
      key = ''
      while key not in env.keys:
        refresh()
        key = input("press key\n$>")
        if key == "exit()":
          exit()
        if (key == 'p'):
          env.show(n_pts=10000)
        if (key == 'v'):
          plt.plot(v); plt.show()
      for _ in range(1):
        _, _, done, _ = env.step_via_key(key)
        v.append(env.state[1])
    again = input("episode done, continue? (Y / n)")
    if again == 'n':
      break


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('-e', '--env', type=str, default='mountain_car',
                      help='Env to play with.',
                      choices=ENV_DICT.keys())
  args = parser.parse_args()
  play(ENV_DICT[args.env])


if __name__ == "__main__":
  main()
