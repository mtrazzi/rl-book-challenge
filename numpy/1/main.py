from agents import RLAgent
from board import TicTacToeBoard
import numpy as np
from utils import play_against_agent


def main():
  np.random.seed(42)
  board = TicTacToeBoard()
  agent = RLAgent()
  agent.train(board, n_episodes=10000)
  while True:
    play_against_agent(board, agent)


if __name__ == '__main__':
  main()
