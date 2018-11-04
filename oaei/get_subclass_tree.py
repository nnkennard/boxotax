import sys
import rdflib
import ontospy

class RDFPredicates(object):
  SYNTAX_FIRST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#first"
  SYNTAX_REST = "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest"
  SYNTAX_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
  DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
  LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
  RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
  SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
  DISJOINT_WITH = "http://www.w3.org/2002/07/owl#disjointWith"
  HAS_VALUE = "http://www.w3.org/2002/07/owl#hasValue"
  MIN_CARDINALITY = "http://www.w3.org/2002/07/owl#minCardinality"
  ON_PROPERTY = "http://www.w3.org/2002/07/owl#onProperty"
  ONE_OF = "http://www.w3.org/2002/07/owl#oneOf"
  UNION_OF = "http://www.w3.org/2002/07/owl#unionOf"

class OAEINode(object):
  def __init__(self, uid, label):
    self.uid = uid
    self.label = label
    self.subclasses = []
    pass

def main():
  owl_file = sys.argv[1]
  g = rdflib.Graph()
  result = g.parse(owl_file)
  oaei_nodes = {}

  for subj, pred, obj in g:
    if str(pred) == RDFPredicates.LABEL:
      oaei_nodes[str(subj)] = OAEINode(str(subj), str(obj))

  for subj, pred, obj in g:
    if str(pred) == RDFPredicates.SUBCLASS_OF:
      if str(subj) in oaei_nodes and str(obj) in oaei_nodes:
        oaei_nodes[str(obj)].subclasses.append(oaei_nodes[str(subj)])

  all_nodes = set([y.uid for y in oaei_nodes.values()])
  subclass_nodes = set([y.uid
    for y in sum([x.subclasses for x in oaei_nodes.values()], [])])
  non_subclass_nodes = all_nodes - subclass_nodes

  for node_id, node in oaei_nodes.items():
    for subclass in node.subclasses:
      print(node.uid+"\t"+subclass.uid)

if __name__ == "__main__":
  main()
