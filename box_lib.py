import collections
import rdflib

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

# UMLS datasets
GO = "go"
HPO = "hpo"
MSH = "msh"

SAB_MAP = {"msh": "MSH",
    "go": "GO",
    "hpo": "HPO"}

PREFIX_MAP =  {"msh": "MSH:",
    "go": "GO:",
    "hpo": "HPO:"}


# Largebio datasets
FMA = "fma"
NCI = "nci"
SNOMED = "snomed"

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
    FMA: LabelPrefix.FMA,
    NCI: LabelPrefix.NCI,
    SNOMED: LabelPrefix.SNOMED,
    }

START_INDEX ={
    NCI: 10000,
    FMA: 100000,
    SNOMED: 200000,
    GO: 100000,
    HPO: 200000,
    MSH: 500000
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
      return label[len(prefix):]


PAIR_TO_DATASET_NAMES = {
    "fma2nci": ["fma", "nci"],
    "fma2snomed": ["fma", "snomed"],
    "snomed2nci": ["nci", "snomed"]
    }

ROOT_STR = "!!ROOT"
#ROOT_IDX_STR = "0"
ROOT_IDX_STR = "000000"

def get_subclass_graph_from_owl_file(owl_file):
  g = rdflib.Graph()
  result = g.parse(owl_file)
  graph = collections.defaultdict(set)

  for subj, pred, obj in g:
    if str(pred) == RDFPredicates.SUBCLASS_OF:
      if is_valid_label(subj) and is_valid_label(obj):
        graph[strip_prefix(obj)].add(strip_prefix(subj))

  return graph


def get_transitive_closure(graph, root, ancestor_list, seen, edges):
  if root in seen:
    return
  seen.append(root)
  for child in sorted(graph[root]):
    get_transitive_closure(graph, child, ancestor_list+[root], seen, edges)
  for ancestor in ancestor_list:
    edges.append((ancestor, root))

