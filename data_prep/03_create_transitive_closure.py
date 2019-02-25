import collections
import sys

import box_lib

def main():
  input_pair_file = sys.argv[1]
  assert input_pair_file.endswith('.pairs')
  graph = collections.defaultdict(list)

  with open(input_pair_file, 'r') as f:
    for line in f:
      parent, child = line.strip().split("\t")
      graph[parent].append(child)

  print(graph)

  transitive_edges = []

  box_lib.get_transitive_closure(
      graph, box_lib.ROOT_IDX_STR, [], [], transitive_edges)


  with open(input_pair_file.replace("pairs", "tpairs"), 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([parent, child]) + "\n")


if __name__ == "__main__":
  main()
