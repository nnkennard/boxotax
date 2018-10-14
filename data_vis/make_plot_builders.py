import sqlite3
import collections
import umls_lib

def hypernym_breakdowns():
  pass

def overlap_percentages():
  pass

def join_rels(relation, fine_relation):
  return relation + "_" + fine_relation

class MultiGraph(object):
  def __init__(self, extended_pairs):
    self.edges_dict = collections.defaultdict(
        lambda:collections.defaultdict(list))
    for (source_node, target_node, relation, fine_relation,
         source) in extended_pairs:
      self.edges_dict[source][join_rels(relation,
        fine_relation)].append((source_node, target_node))

    self.graphs_dict = collections.defaultdict(dict)
    for source, rels in self.edges_dict.iteritems():
      for rel_pair, pairs in rels.iteritems():
        self.graphs_dict[source][rel_pair] = SimpleGraph(source, rel_pair,
            pairs)

    self.hypernym_percents_dict = self.hypernym_breakdowns()
    self.overlap_percents_dict = self.overlap_percentages()

  def hypernym_breakdowns(self):
    counts_dict = collections.defaultdict(dict)
    percents_dict = collections.defaultdict(dict)
    for source, rels in self.edges_dict.iteritems():
      for rel_pair, pairs in rels.iteritems():
        counts_dict[source][rel_pair] = len(pairs)
      source_total = float(sum(counts_dict[source].values()))
      for rel_pair, pairs in rels.iteritems():
        percents_dict[source][rel_pair] = float(len(pairs))/source_total
        print source, rel_pair, percents_dict[source][rel_pair]
      print
    return percents_dict

  def overlap_percentages(self):
    overlap_percentage_dict = collections.defaultdict(dict)
    pairs_by_source = dict()
    for source, rels in self.edges_dict.iteritems():
      pairs_by_source[source] = set(sum(rels.values(), []))

    for source_1, pairs_1 in pairs_by_source.iteritems():
      for source_2, pairs_2 in pairs_by_source.iteritems():
        overlap_percentage_dict[source_1][source_2] = len(
            pairs_1.intersection(pairs_2))/float(len(pairs_1))
        print source_1, source_2, overlap_percentage_dict[source_1][source_2]

    return overlap_percentage_dict


class SimpleGraph(object):
  def __init__(self, source, relations, pairs):
    self.source = source
    self.relations = relations

    source_nodes, target_nodes = zip(*pairs)
    self.subroot_identifiers = list(set(source_nodes) - set(target_nodes))

    node_identifiers = sorted(list(set(source_nodes).union(set(target_nodes))))
    self.nodes = {identifier:SimpleNode(identifier) for identifier in
        node_identifiers}

    self.build(pairs)

  def build(self, pairs):
    super_root = SimpleNode("ROOT", None)
    for identifier in self.subroot_identifiers:
      super_root.children.append(self.nodes[identifier])
    for source, target in pairs:
      self.nodes[source].children.append(self.nodes[target])
    return super_root

  def connected_components(self):
    return self.root.children.values()


class SimpleNode(object):
  def __init__(self, identifier, text=None):
    self.identifier = identifier
    self.text = text
    self.children = []


def main():
  config_line = "config_msh_mth	CHD|RN	None	None	None	None"
  conn = sqlite3.connect(umls_lib.DB_FILE)
  c = conn.cursor()

  subgraph = umls_lib.read_subgraph_from_config_line(config_line)
  pairs = subgraph.get_pairs(conn)

  source = "ALL"
  relations = "PAR|BR"
  g = MultiGraph(pairs)
  pass

if __name__ == "__main__":
  main()
