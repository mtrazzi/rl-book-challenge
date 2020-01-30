# for exercise 2.7


def weights(i, n, beta=0.5):
  return (((1-beta) ** 2) * (beta ** (n-i)) * (1 / (1 - beta ** i)) *
          (1 / (1 - beta ** (n-i+1))))


def sum_weights(n, beta=0.8):
  return sum([weights(i, n, beta) for i in range(1, n+1)])


def main():
  print(sum_weights(1000))


if __name__ == '__main__':
  main()
