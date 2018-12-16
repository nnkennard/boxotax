import pickle
import sys
import random
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

def get_negative_edges(transitive_edges, sorted_nodes):
  negative_edges = []
  while len(negative_edges) < len(transitive_edges):
    possible_edge = tuple(random.sample(sorted_nodes, 2))
    if (possible_edge not in transitive_edges
        and possible_edge not in negative_edges):
      negative_edges.append(possible_edge)
  return negative_edges


def main():
  random.seed(43)
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

  # TODO: this, but less horribly
  prefixed_root = list(all_nodes)[0].split('_')[0] + "_" + ROOT_STR
  graph[prefixed_root] = list(non_subclass_nodes)
  all_nodes.add(prefixed_root)

  sorted_nodes = sorted(all_nodes)


  transitive_edges = []
  get_transitive_closure(graph, prefixed_root, [], [], transitive_edges)
  negative_edges = get_negative_edges(transitive_edges, sorted_nodes)


  unary_filename = owl_file.replace(".owl", ".unary")
  with open(unary_filename, 'w') as f:
    for node in sorted_nodes:
      f.write(node + "\n")

  pairwise_filename = owl_file.replace(".owl", ".out")
  with open(pairwise_filename, 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([parent, child, "true"])+"\n")
    for parent, child in negative_edges:
      f.write("\t".join([parent, child, "false"])+"\n")


if __name__ == "__main__":
  main()
