DB_FILE = "/Users/nnayak/git_repos/boxotax/umls/umls.db"

CONCEPT_FILE = "META/MRCONSO.RRF"
RELATION_FILE = "META/MRREL.RRF"
SEMTYPE_FILE = "META/MRSTY.RRF"

ALL_RELATION_TYPES = ["RO", "RB", "RN", "PAR", "CHD"]

class SemanticTypePaths(object):
  FISH = "A1.1.3.1.1.3"
  DISEASE = "B2.2.1.2.1"
  DRUG = "A1.3.1.1"
  GENE = "A1.2.3.5"
  ENTITY = "A"
  EVENT = "B"

SEMANTIC_TYPE_NAMES = {
  SemanticTypePaths.FISH : "Fish",
  SemanticTypePaths.DISEASE : "Disease",
  SemanticTypePaths.DRUG : "Drug",
  SemanticTypePaths.GENE : "Gene",
  SemanticTypePaths.ENTITY : "Entity",
  SemanticTypePaths.EVENT : "Event"
}


IMPORTANT_TYPE_PATHS = [SemanticTypePaths.FISH, SemanticTypePaths.DISEASE,
    SemanticTypePaths.DRUG, SemanticTypePaths.GENE, SemanticTypePaths.EVENT,
    SemanticTypePaths.ENTITY]


ALL_SOURCES = ["AIR", "AOD", "AOT", "ATC" "CCS", "CCS_10", "CSP", "CST",
"CVX", "FMA", "GO", "HCPCS", "HGNC", "HL7V2.5", "HL7V3.0", "HPO", "ICD10PCS",
"ICD9CM", "ICPC", "LCH_NW", "LNC", "MEDLINEPLUS", "MSH", "MTH", "MTHHH",
"MTHICD9", "MTHMST", "MTHSPL", "MVX", "NCBI",  "NCI", "NDFRT", "NDFRT_FDASPL",
"NDFRT_FMTSME", "OMIM", "PDQ", "RAM", "RXNORM", "SOP", "SRC", "TKMT", "USPMG",
"UWDA", "VANDF"]


def read_file(filename):
  with open(filename, 'r') as f:
    for line in f:
      fields = line.strip().split('|')
      yield fields

def comma_join_strings(items):
  return ", ".join("'" + item + "'" for item in items)

def read_config(relation_types_string, fine_relation_types_string,
    included_sources_string, excluded_sources_string):
  return [x.split("|") if x != "None" else None
      for x in [relation_types_string, fine_relation_types_string,
        included_sources_string,excluded_sources_string]]

def read_subgraph_from_config_line(config_line):
  fields = config_line.strip().split("\t")
  (relation_types, fine_relation_types, included_sources,
      excluded_sources) = read_config(*fields[1:5])

  if fields[5] != "None":
    type_prefix = fields[5]
  else:
    type_prefix = ""
  return UMLSSubgraph(relation_types, fine_relation_types,
      included_sources, excluded_sources, None, type_prefix)


class UMLSSubgraph(object):
  def __init__(self, relation_types, fine_relation_types, included_sources,
      excluded_sources, subgraph_name, type_prefix=""):
    """
      Args:
        relation_types: a list of relation types. If None, all relations are
        included
        fine_relation_types: at most one fine relation type; overrides
        relation_types.
        included_sources: a list of sources to be included. If None, all sources are
        included.
        excluded_sources: a list of sources to be excluded. If None, ignored
        subgraph_name: a name for the subgraph in the db
        type_prefix: semantic type as a prefix of the semantic tree type
    """

    # TODO: Change the 'IN's to 'LIKE's when possible
    if fine_relation_types is not None:
      assert relation_types is None
    self.fine_relation_types = fine_relation_types
    if relation_types is None:
      self.relation_types = ALL_RELATION_TYPES
    else:
      self.relation_types = relation_types

    if included_sources is None:
      included_sources = ALL_SOURCES
    if excluded_sources is None:
      self.sources = set(included_sources)
    else:
      self.sources = set(included_sources) - set(excluded_sources)

    self.type_prefix = type_prefix

  def get_pairs(self, conn):
    """
      Args:
        conn: Database connection
    """

    if self.type_prefix == "":
      type_join_string = ""
      type_filter = ""
    else:
      type_join_string = ("LEFT JOIN types as t1 on relations.cui1 = t1.cui "
                          "LEFT JOIN types as t2 on relations.cui2 = t2.cui")
      type_filter = ("AND t1.type_tree_number LIKE '{}%' "
                     "AND t2.type_tree_number LIKE '{}%'").format(
                         self.type_prefix, self.type_prefix)

    select_string = ("SELECT DISTINCT cui1, cui2, relation, "
                     "fine_relation, source FROM relations {} "
                     "WHERE ").format(type_join_string)
    select_string += "source IN ({})".format(
      comma_join_strings(self.sources))
    if self.fine_relation_types is not None:
      select_string += " AND fine_relation IN ({})".format(
        comma_join_strings(self.fine_relation_types))
    else:
      assert (self.fine_relation_types is None
              and self.relation_types is not None)
      select_string += " AND relation IN ({}) {}".format(
        comma_join_strings(self.relation_types), type_filter)
 
    c = conn.cursor()
    c.execute(select_string)
    return c.fetchall()
    
