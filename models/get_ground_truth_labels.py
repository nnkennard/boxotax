import sys
import collections
import box_lib

def maybe_hypo(x):
  return box_lib.label(x) == box_lib.HYPO

def maybe_hyper(x):
  return box_lib.label(x) == box_lib.HYPER

def maybe_unrelated(x):
  return box_lib.label(x) == box_lib.UNRELATED

aHYPO, aHYPER, aUNRELATED, aALIGNED, aERROR = range(5)

def main():
  cond_probs = collections.defaultdict(lambda:collections.defaultdict())
  with open(sys.argv[1], 'r') as f:
    for line in f:
      ind_1, ind_2, prob = line.strip().split()
      cond_probs[ind_1][ind_2] = float(prob)

  for ind_1, probs in cond_probs.items():
    for ind_2, prob in probs.iteritems():
      rev_prob = cond_probs[ind_2][ind_1]
      if maybe_hypo(prob) and maybe_hyper(rev_prob):
        print aHYPO
      elif maybe_hyper(prob) and maybe_hypo(rev_prob):
        print aHYPER
      elif maybe_hyper(prob) and maybe_hyper(rev_prob):
        print aALIGNED
      elif maybe_unrelated(prob) and maybe_unrelated(rev_prob):
        print aUNRELATED
      else:
        print aERROR


if __name__ == "__main__":
  main()
