import pickle
import sys
import rdflib
import oaei_lib
import collections

# TODO: WRITE A TEST FOR THIS OMG

ROOT_STR = "!!ROOT"
# I just want it to get index 0

def get_transitive_closure(graph, root, ancestor_list, seen, edges):
  if root in seen:
    return
  seen.append(root)
  for child in sorted(graph[root]):
    get_transitive_closure(graph, child, ancestor_list+[root], seen, edges)
  for ancestor in ancestor_list:
    edges.append((ancestor, root))

def main():
  owl_file = sys.argv[1]

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

  graph[ROOT_STR] = list(non_subclass_nodes)
  all_nodes.add(ROOT_STR)

  sorted_nodes = sorted(all_nodes)
  node_to_index = {node:str(i) for i, node in enumerate(sorted_nodes)}

  transitive_edges = []
  get_transitive_closure(graph, ROOT_STR, [], [], transitive_edges)

  unary_filename = owl_file.replace(".owl", ".unary")
  with open(unary_filename, 'w') as f:
    for node in sorted_nodes:
      f.write("\t".join([node, node_to_index[node]]) + "\n")

  pairwise_filename = owl_file.replace(".owl", ".out")
  with open(pairwise_filename, 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([node_to_index[parent], node_to_index[child],
        parent, child])+"\n")


if __name__ == "__main__":
  main()
