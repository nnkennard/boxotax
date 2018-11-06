import pickle
import sys
import rdflib
import oaei_lib

def main():
  owl_file = sys.argv[1]
  output_file = owl_file + ".graph.pkl"
  g = rdflib.Graph()
  result = g.parse(owl_file)
  oaei_nodes = {}

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.LABEL:
      oaei_nodes[str(subj)] = oaei_lib.OAEINode(str(subj), str(obj))

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.SUBCLASS_OF:
      if str(subj) in oaei_nodes and str(obj) in oaei_nodes:
        oaei_nodes[str(obj)].subclasses.append(str(subj))

  all_nodes = set(oaei_nodes.keys())
  subclass_nodes = set(sum([x.subclasses for x in oaei_nodes.values()], []))
  non_subclass_nodes = all_nodes - subclass_nodes

  root_node = oaei_lib.OAEINode("ROOT", "ROOT")
  root_node.subclasses = list(non_subclass_nodes)
  oaei_nodes["ROOT"] = root_node

  graph = oaei_lib.OAEIGraph(oaei_nodes, root_node)
  with open(output_file, 'wb') as f:
    pickle.dump(graph, f)


if __name__ == "__main__":
  main()
