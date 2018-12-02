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

class LabelPrefix(object):
  FMA = "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"
  NCI = "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"
  SNOMED = "http://www.ihtsdo.org/snomed#"

LABEL_PREFIX_MAP ={
    "fma": LabelPrefix.FMA,
    "nci": LabelPrefix.NCI,
    "snomed": LabelPrefix.SNOMED,
    }

ENTITY_1 = "http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity1"
ENTITY_2 = "http://knowledgeweb.semanticweb.org/heterogeneity/alignmententity2"

PAIR_LABELS = {
    ENTITY_1 : "entity1",
    ENTITY_2 : "entity2"
    }

def is_valid_label(label):
  checks = set([label.startswith(x) for x in LABEL_PREFIX_MAP.values()])
  return checks == set(([False, False, True]))

def strip_prefix(label):
  for dataset, prefix in LABEL_PREFIX_MAP.items():
    if label.startswith(prefix):
      return dataset + "_" +label[len(prefix):]


PAIR_TO_DATASET_NAMES = {
    "fma2nci": ["fma", "nci"],
    "fma2snomed": ["fma", "snomed"],
    "snomed2nci": ["nci", "snomed"]
    }


