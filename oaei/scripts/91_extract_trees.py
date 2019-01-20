import collections
import random
import rdflib
import sys

import oaei_lib

def main():
  random.seed(43)
  owl_file, dataset = sys.argv[1:3]

  g = rdflib.Graph()
  result = g.parse(owl_file)
  graph = collections.defaultdict(set)

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.SUBCLASS_OF:
      if oaei_lib.is_valid_label(subj) and oaei_lib.is_valid_label(obj):
        graph[oaei_lib.strip_prefix(obj)].add(oaei_lib.strip_prefix(subj))

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(set.union(*graph.values()))
  all_nodes = superclass_nodes.union(subclass_nodes)
  non_subclass_nodes = all_nodes - subclass_nodes

  graph[oaei_lib.ROOT_STR] = list(non_subclass_nodes)
  node_to_index = {node : str(oaei_lib.START_INDEX[dataset] + i) for
      i, node in enumerate(sorted(all_nodes))}
  node_to_index[oaei_lib.ROOT_STR] = "0"
  index_to_node = {index: node for node, index in node_to_index.items()}

  for parent, children in graph.items():
    parent_idx = node_to_index[parent]
    for child in children:
      print("\t".join([parent_idx, node_to_index[child]]))

  for index in sorted(index_to_node.keys()):
    print("\t".join([index, index_to_node[index]]))


if __name__ == "__main__":
  main()
