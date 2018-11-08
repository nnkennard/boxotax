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
    self.parent = None
  
  def set_parent(self, parent):
    self.parent = parent

class OAEIGraph(object):
  def __init__(self, nodes, root):
    self.nodes = nodes
    self.root = root
    pass

