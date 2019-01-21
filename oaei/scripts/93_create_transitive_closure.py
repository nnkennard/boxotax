import collections
import sys

import oaei_lib

def main():
  input_pair_file = sys.argv[1]
  graph = collections.defaultdict(list)

  with open(input_pair_file, 'r') as f:
    for line in f:
      parent, child = line.strip().split("\t")
      graph[parent].append(child)

  transitive_edges = []

  oaei_lib.get_transitive_closure(
      graph, oaei_lib.ROOT_IDX_STR, [], [], transitive_edges)


  with open(input_pair_file.replace("pairs", "tpairs"), 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([parent, child]) + "\n")



if __name__ == "__main__":
  main()
