import sys
import umls_lib
import sqlite3

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
      included_sources, excluded_sources, type_prefix)


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
      self.relation_types = umls_lib.ALL_RELATION_TYPES
    else:
      self.relation_types = relation_types

    if included_sources is None:
      included_sources = umls_lib.ALL_SOURCES
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
    select_string = "SELECT DISTINCT cui1, cui2, relation, fine_relation, source FROM RELATIONS WHERE "
    select_string += "source IN ({})".format(
      comma_join_strings(self.sources))
    if self.fine_relation_types is not None:
      select_string += " AND fine_relation IN ({})".format(
        comma_join_strings(self.fine_relation_types))
    else:
      assert (self.fine_relation_types is None
              and self.relation_types is not None)
      select_string += " AND relation IN ({})".format(
        comma_join_strings(self.relation_types))
    c = conn.cursor()
    c.execute(select_string)
    for i in c.fetchall():
      print "\t".join(list(i))


def main():
  conn = sqlite3.connect(umls_lib.DB_FILE)
  c = conn.cursor()

  with open(sys.argv[1], 'r') as f:
    for line in f:
      subgraph = read_subgraph_from_config_line(line)
      subgraph.get_pairs(conn)
  pass

if __name__=="__main__":
  main()
