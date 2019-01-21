import collections
import sys

def main():
  cond_probs = collections.defaultdict(dict)
  with open(sys.argv[1], 'r') as f:
    for line in f:
      a, b, prob = line.strip().split()
      cond_probs[a][b] = prob

  with open(sys.argv[1] + ".final", 'w') as f:
    for a, probs in cond_probs.items():
      for b, prob in probs.items():
        if int(a) < int(b):
          f.write("\t".join([a, b, prob, cond_probs[b][a]]))
          f.write("\n")

if __name__ == "__main__":
  main()
