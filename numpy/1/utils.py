import os

from board import TicTacToeBoard
import matplotlib.pyplot as plt
import numpy as np


def opposite(sym):
  return 'x' if sym == 'o' else 'o'


def play_against_agent(agent):
  board = TicTacToeBoard(size=agent.size)
  while True:
    board.reset()
    while not board.is_end_state():
      os.system('cls' if os.name == 'nt' else 'clear')
      print(board)
      x, y = [int(elt) for elt in input().split()]
      board.do_move(x, y)
      print(board)
      print(agent.get_possible_move_values(board))
      input("...")
      board.do_move(*agent.best_move(board))


def test_agent(agent, opponent, n_episodes=10000):
  from agents import RLAgent
  board = TicTacToeBoard(agent.size)
  n_wins = 0
  for _ in range(n_episodes):
    board.reset()
    counter = 0
    while not board.is_end_state():
      to_play = opponent if counter % 2 == 0 else agent
      if isinstance(to_play, RLAgent):
        board.do_move(*(to_play.eps_greedy(board)[1]))
      else:
        board.do_move(*to_play.best_move(board))
      counter += 1
    n_wins += board.has_won(agent.sym)
  print(f"{n_wins}/{n_episodes}")
  return n_wins / n_episodes


def weighted_averages(arr, alpha=0.9):
  new_arr = np.zeros_like(arr)
  avg = None
  for i in range(len(arr)):
    avg = alpha * avg + (1 - alpha) * arr[i] if avg is not None else arr[i]
    new_arr[i] = avg
  return new_arr


def benchmark(agent, opponent, step=10,
              training_steps=100, n_eval_episodes=1000, alpha=0.9):
  results = []
  for _ in range(training_steps // step):
    results.append(test_agent(agent, opponent, n_eval_episodes))
    agent.train(opponent, step)
  plt.plot(weighted_averages(results, alpha))
  plt.show()


def opposite_agent(agent):
  from agents import RLAgent
  """Returns an agent who plays with the opposite symbol."""
  new_agent = RLAgent(agent.size, opposite(agent.sym), agent.step, agent.eps,
                      agent.eps_decay)
  if isinstance(agent, RLAgent):
    for state, val in agent.V.items():
      new_agent.V[state] = 1 - val
  return new_agent
