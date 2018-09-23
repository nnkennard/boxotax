
class UMLSSubgraph(object):
  def __init__(relation_type, fine_relation_type, included_sources,
      excluded_sources, type_prefix=""):
    """
      Args:
        relation_type: at most one relation type
        fine_relation_type: at most one fine relation type; overrides
        relation_type.
        included_sources: a list of sources to be included. If None, all sources are
        included.
        excluded_sources: a list of sources to be excluded. If None, ignored
        type_prefix: semantic type as a prefix of the semantic tree type
    """
    if fine_relation_type is not None:
      assert relation_type is None
      self.fine_relation_type = fine_relation_type
    self.relation_type = relation_type

    if included_sources is None:
      included_sources = umls_lib.all_sources
    if excluded_sources is None:
      sources = set(included_sources)
    else:
      sources = set(included_sources) - set(excluded_sources)

    self.type_prefix = type_prefix  

  def get_pairs(self, conn):


def main():
  pass

if __name__=="__main__":
  main()
