import pickle
import sys
import rdflib
import oaei_lib
import collections

# TODO: WRITE A TEST FOR THIS OMG

class LabelPrefix(object):
  FMA = "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"

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

def unprefix(input_string, prefix):
  assert input_string.startswith(prefix)
  return input_string[len(prefix):]

def main():
  owl_file, output_prefix = sys.argv[1], sys.argv[2]

  g = rdflib.Graph()
  result = g.parse(owl_file)
  graph = collections.defaultdict(set)

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.SUBCLASS_OF:
      if subj.startswith(LabelPrefix.FMA) and obj.startswith(LabelPrefix.FMA):
        s, o = unprefix(str(subj), LabelPrefix.FMA), unprefix(str(obj),
            LabelPrefix.FMA)
        graph[o].add(s)

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

  unary_filename = output_prefix + ".unary"
  with open(unary_filename, 'w') as f:
    for node in sorted_nodes:
      f.write("\t".join([node, node_to_index[node]]) + "\n")

  pairwise_filename = output_prefix + ".pairwise"
  with open(pairwise_filename, 'w') as f:
    for parent, child in transitive_edges:
      f.write("\t".join([node_to_index[parent], node_to_index[child],
        parent, child])+"\n")


if __name__ == "__main__":
  main()
