import pickle
import sys
import rdflib
import oaei_lib


class LabelPrefix(object):
  FMA = "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"

ROOT_STR = "!!ROOT"
# I just want it to get index 0

def BFS(root_oaei_node, oaei_nodes, edges):
  """BFS over the tree, adding (parent, child) pairs to `edges`."""
  own_uid = root_oaei_node.uid
  for subclass in root_oaei_node.subclasses:
    edges.append((own_uid, oaei_nodes[subclass].uid))
  for subclass in root_oaei_node.subclasses:
    BFS(oaei_nodes[subclass], oaei_nodes, edges)


def unprefix(input_string, prefix):
  assert input_string.startswith(prefix)
  return input_string[len(prefix):]

def main():
  owl_file, output_prefix = sys.argv[1], sys.argv[2]

  g = rdflib.Graph()
  result = g.parse(owl_file)
  oaei_nodes = {}

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.LABEL:
      s = unprefix(str(subj), LabelPrefix.FMA)
      oaei_nodes[s] = oaei_lib.OAEINode(s, obj)

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.SUBCLASS_OF:
      if subj.startswith(LabelPrefix.FMA) and obj.startswith(LabelPrefix.FMA):
        s, o = unprefix(str(subj), LabelPrefix.FMA), unprefix(str(obj),
            LabelPrefix.FMA)

        # This should include all nodes that participate in the subclass relation
        # -- still scary though
        if s in oaei_nodes and o in oaei_nodes:
          oaei_nodes[o].subclasses.append(s)

  all_nodes = set(oaei_nodes.keys())
  subclass_nodes = set(sum([x.subclasses for x in oaei_nodes.values()], []))
  non_subclass_nodes = all_nodes - subclass_nodes

  root_node = oaei_lib.OAEINode(ROOT_STR, ROOT_STR)
  root_node.subclasses = list(non_subclass_nodes)
  oaei_nodes[ROOT_STR] = root_node

  sorted_nodes = sorted(oaei_nodes.keys())
  node_to_index = {node:str(i) for i, node in enumerate(sorted_nodes)}

  edges = []
  BFS(root_node, oaei_nodes, edges)
  for parent, child  in edges:
    print("\t".join([node_to_index[parent], node_to_index[child],
      parent, child]))




if __name__ == "__main__":
  main()
