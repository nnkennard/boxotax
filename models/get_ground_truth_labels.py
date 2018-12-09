import sys
import collections
import box_lib

def maybe_hypo(x):
  return box_lib.label(x) == box_lib.HYPO

def maybe_hyper(x):
  return box_lib.label(x) == box_lib.HYPER

def maybe_unrelated(x):
  return box_lib.label(x) == box_lib.UNRELATED

def main():
  cond_probs = collections.defaultdict(lambda:collections.defaultdict())
  with open(sys.argv[1], 'r') as f:
    for line in f:
      ind_1, ind_2, prob = line.strip().split()
      cond_probs[ind_1][ind_2] = float(prob)

  for ind_1, probs in cond_probs.items():
    for ind_2, prob in probs.iteritems():
      rev_prob = cond_probs[ind_2][ind_1]
      if maybe_hypo(prob):
        assert maybe_hypo(rev_prob) or maybe_hyper(rev_prob)
      elif maybe_hyper(prob):
        if not maybe_hypo(rev_prob):
          #print ind_1, ind_2, prob, rev_prob
          continue
          assert maybe_hyper(rev_prob)
      elif maybe_unrelated(prob):
        if not maybe_unrelated(rev_prob):
          pass
      else:
        assert maybe_unrelated(rev_prob)

if __name__ == "__main__":
  main()
