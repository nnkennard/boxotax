import collections
import random
import sys

import oaei_lib

def main():
  random.seed(43)
  input_pair_file = sys.argv[1]

  graph = collections.defaultdict(list)
  node_set = set()
  with open(input_pair_file, 'r') as f:
    for line in f:
      parent, child = line.strip().split("\t")
      graph[parent].append(child)
      node_set.add(parent)
      node_set.add(child)

  node_list = sorted(node_set)

  transitive_edges = []
  oaei_lib.get_transitive_closure(
      graph, oaei_lib.ROOT_IDX_STR, [], [], transitive_edges)

  negative_ratio = 1.0
  total_negative_edges = negative_ratio * len(transitive_edges)

  negative_edges = []
  while len(negative_edges) < total_negative_edges:
    if len(negative_edges) % 100 == 0:
      print(str(len(negative_edges)) + "/" + str(total_negative_edges))
    parent = random.choice(node_list)
    child = random.choice(node_list)
    if parent == child:
      continue
    elif (parent, child) in transitive_edges:
      continue
    else:
      negative_edges.append((parent, child))

  output_file = input_pair_file.replace(".no_neg", "")

  with open(output_file, 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([parent, child, "1"]) + "\n")
    for parent, child in negative_edges:
      f.write("\t".join([parent, child, "0"]) + "\n")




if __name__ == "__main__":
  main()

