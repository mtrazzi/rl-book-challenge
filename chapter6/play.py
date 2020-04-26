import argparse
import os
from windy_gridworld import WindyGridworld
from cliff import TheCliff

ENV_DICT = {
  'windy_gridworld': WindyGridworld(),
  'cliff': TheCliff(),
}

def play(env):
  def refresh():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(env)
  while True:
    s = env.reset()
    done = False
    while not done:
      key = ''
      while key not in env.keys:
        refresh()
        key = input("press key\n$>")
        if key == "exit()":
          exit()
      _, _, done, _ = env.step_via_key(key)
    again = input("episode done, continue? (Y / n)")
    if again == 'n':
      break

def main():
  parser = argparse.ArgumentParser()

  parser.add_argument('env', type=str, default=None,
                      help='Env to play with.',
                      choices=ENV_DICT.keys())
  args = parser.parse_args()
  play(ENV_DICT[args.env])

if __name__ == "__main__":
  main()
