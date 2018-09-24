import umls_lib
import sqlite3

def comma_join_strings(items):
  return ", ".join("'" + item + "'" for item in items)

class UMLSSubgraph(object):
  def __init__(self, relation_types, fine_relation_types, included_sources,
      excluded_sources, type_prefix=""):
    """
      Args:
        relation_types: at most one relation type
        fine_relation_types: at most one fine relation type; overrides
        relation_types.
        included_sources: a list of sources to be included. If None, all sources are
        included.
        excluded_sources: a list of sources to be excluded. If None, ignored
        type_prefix: semantic type as a prefix of the semantic tree type
    """
    if fine_relation_types is not None:
      assert relation_types is None
    self.fine_relation_types = fine_relation_types
    self.relation_types = relation_types

    if included_sources is None:
      included_sources = umls_lib.all_sources
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
    select_string = "SELECT DISTINCT cui1, cui2 FROM RELATIONS WHERE "
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
    print(select_string)
    c = conn.cursor()
    c.execute(select_string)
    for i in c.fetchall():
      print i


def main():
  conn = sqlite3.connect(umls_lib.DB_FILE)
  c = conn.cursor()

  subgraph = UMLSSubgraph(relation_types=['PAR'],
                          fine_relation_types=None,
                          included_sources=None,
                          excluded_sources=["MTH"])
  subgraph.get_pairs(conn)

  #c.execute(
  #"SELECT DISTINCT cui1, cui2 FROM relations WHERE source in ('MSH', 'RXNORM') AND relation in ('PAR')")
  #for i in c.fetchall():
  #  print i
  conn.close()

  pass

if __name__=="__main__":
  main()
