import collections
import random
import rdflib
import sys

import oaei_lib

def get_subclass_graph_from_text_file(text_file):
  graph = collections.defaultdict(set)
  with open(text_file, 'r') as f:
    for line in f:
      child, parent = line.strip().split("\t")
      graph[parent].add(child)
  return graph


def main():
  random.seed(43)
  text_file, dataset = sys.argv[1:3]

  graph = get_subclass_graph_from_text_file(text_file)

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(set.union(*graph.values()))
  all_nodes = superclass_nodes.union(subclass_nodes)
  non_subclass_nodes = all_nodes - subclass_nodes
  graph[oaei_lib.ROOT_STR] = list(non_subclass_nodes)

  node_to_index = {node : str(oaei_lib.START_INDEX[dataset] + i) for
      i, node in enumerate(sorted(all_nodes))}
  node_to_index[oaei_lib.ROOT_STR] = "0"
  index_to_node = {index: node for node, index in node_to_index.items()}

  with open(text_file.replace("txt", "pairs"), 'w') as f:
    for parent, children in graph.items():
      parent_idx = node_to_index[parent]
      for child in children:
        f.write("\t".join([parent_idx, node_to_index[child]]) + "\n")

  with open(text_file.replace("txt", "vocab"), 'w') as f:
    for index in sorted(index_to_node.keys()):
      f.write("\t".join([index, index_to_node[index]]) + "\n")


if __name__ == "__main__":
  main()
