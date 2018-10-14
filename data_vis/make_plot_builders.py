import sqlite3
import collections
import umls_lib

def hypernym_breakdowns():
  pass

def overlap_percentages():
  pass

SUPER_ROOT = "super_root_child"

class SimpleGraph(object):
  def __init__(self, source, relations, pairs):
    self.source = source
    self.relations = relations

    sources, targets, relations, _, _ = zip(*pairs)
    self.subroot_identifiers = list(set(sources) - set(targets))
    node_identifiers = sorted(list(set(sources).union(set(targets))))

    self.nodes = {identifier:SimpleNode(identifier) for identifier in
        node_identifiers}

    self.root = self.build_graph(zip(sources, targets, relations))

  def build_graph(self, pairs):
    super_root = SimpleNode("ROOT", None)
    for identifier in self.subroot_identifiers:
      super_root.children[SUPER_ROOT].append(self.nodes[identifier])
    for source, target, relation_type in pairs:
      self.nodes[source].children[relation_type].append(self.nodes[target])
    return super_root

  def connected_components(self):
    return self.root.children.values()


class SimpleNode(object):
  def __init__(self, identifier, text=None):
    self.identifier = identifier
    self.text = text
    self.children = collections.defaultdict(list)


def main():
  config_line = "config_msh_mth	CHD|RN	None	None	None	A1.1.3.1.1.3"
  conn = sqlite3.connect(umls_lib.DB_FILE)
  c = conn.cursor()

  subgraph = umls_lib.read_subgraph_from_config_line(config_line)
  pairs = subgraph.get_pairs(conn)

  source = "ALL"
  relations = "PAR|BR"
  g = SimpleGraph(source, relations, pairs)
  pass

if __name__ == "__main__":
  main()
