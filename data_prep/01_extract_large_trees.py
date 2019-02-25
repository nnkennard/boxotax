import collections
import random
import rdflib
import sys

import box_lib

def main():
  random.seed(43)
  owl_file, dataset = sys.argv[1:3]

  graph = box_lib.get_subclass_graph_from_owl_file(owl_file)

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(set.union(*graph.values()))
  all_nodes = superclass_nodes.union(subclass_nodes)
  non_subclass_nodes = all_nodes - subclass_nodes
  graph[box_lib.ROOT_STR] = list(non_subclass_nodes)

  node_to_index = {node : str(box_lib.START_INDEX[dataset] + i) for
      i, node in enumerate(sorted(all_nodes))}
  node_to_index[box_lib.ROOT_STR] = "0"
  index_to_node = {index: node for node, index in node_to_index.items()}

  with open(owl_file.replace("owl", "pairs"), 'w') as f:
    for parent, children in graph.items():
      parent_idx = node_to_index[parent]
      for child in children:
        f.write("\t".join([parent_idx, node_to_index[child]]) + "\n")

  with open(owl_file.replace("owl", "vocab"), 'w') as f:
    for index in sorted(index_to_node.keys()):
      f.write("\t".join([index, index_to_node[index]]) + "\n")


if __name__ == "__main__":
  main()
