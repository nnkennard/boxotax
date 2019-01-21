import random
import rdflib
import sys

import oaei_lib

def get_vocab_from_file(vocab_file):
  vocab_dict = {}
  with open(vocab_file, 'r') as f:
    for line in f:
      idx, name = line.strip().split()
      vocab_dict[name] = idx
  return vocab_dict


def main():
  random.seed(43)
  large_vocab_file, small_owl_file, dataset = sys.argv[1:4]
  large_vocab = get_vocab_from_file(large_vocab_file)

  graph = oaei_lib.get_subclass_graph_from_owl_file(small_owl_file)

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(set.union(*graph.values()))
  all_nodes = superclass_nodes.union(subclass_nodes)
  non_subclass_nodes = all_nodes - subclass_nodes
  graph[oaei_lib.ROOT_STR] = list(non_subclass_nodes)

  with open(small_owl_file.replace("owl", "pairs"), 'w') as f:
    for parent, children in graph.items():
      parent_idx = large_vocab[parent]
      for child in children:
        f.write("\t".join([parent_idx, large_vocab[child]]) + "\n")






if __name__ == "__main__":
  main()
