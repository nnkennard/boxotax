import operator
import sqlite3
import collections
import umls_lib
import numpy as np
import matplotlib.pyplot as plt

def join_rels(relation, fine_relation):
  return relation + "_" + fine_relation

def make_bar_list(all_rel_pairs, rel_to_percent_dict):
  return [rel_to_percent_dict[rel_pair] for rel_pair in all_rel_pairs]

class MultiGraph(object):
  def __init__(self, extended_pairs):
    self.edges_dict = collections.defaultdict(
        lambda:collections.defaultdict(list))
    for (source_node, target_node, relation, fine_relation,
         source) in extended_pairs:
      self.edges_dict[source][join_rels(relation,
        fine_relation)].append((source_node, target_node))

    self.graphs_dict = collections.defaultdict(dict)
    for source, rels in self.edges_dict.items():
      for rel_pair, pairs in rels.items():
        self.graphs_dict[source][rel_pair] = SimpleGraph(source, rel_pair,
            pairs)

    #self.overlap_percents_dict = self.overlap_percentages()

  def hypernym_breakdowns(self, name):
    counts_dict = collections.defaultdict(dict)
    percents_dict = collections.defaultdict(lambda
        :collections.defaultdict(float))
    rel_pairs = set()
    sources = set()
    for source, rels in self.edges_dict.items():
      sources.add(source)
      for rel_pair, pairs in rels.items():
        rel_pairs.add(rel_pair)
        counts_dict[source][rel_pair] = len(pairs)
      source_total = float(sum(counts_dict[source].values()))
      for rel_pair, pairs in rels.items():
        percents_dict[source][rel_pair] = float(len(pairs))/source_total
        print(source, rel_pair, percents_dict[source][rel_pair])
      print()

    all_rel_pairs = sorted(list(rel_pairs))
    all_sources = sorted(list(sources))
    ind = np.arange(len(all_sources))

    cumulative = [0.0] * len(all_sources)

    legend_builder = []
    for rel_pair in all_rel_pairs:
      print(rel_pair)
      k = [percents_dict[source][rel_pair] for source in all_sources]
      print(all_rel_pairs, len(all_rel_pairs))

      p = plt.bar(ind, k, bottom=cumulative)
      legend_builder.append(p[0])
      cumulative = [sum(x) for x in zip(cumulative, k)]
      print(k, len(k))
    plt.ylabel('Scores')
    plt.title('Hypernym breakdown for '+ name)
    plt.xticks(ind, all_sources, rotation='vertical')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.legend(legend_builder, all_rel_pairs)

    plt.savefig("/Users/nnayak/hypernym_breakdown_"+name+".png")
    plt.clf()
    return percents_dict

  def overlap_percentages(self):
    overlap_percentage_dict = collections.defaultdict(dict)
    pairs_by_source = dict()
    for source, rels in self.edges_dict.items():
      pairs_by_source[source] = set(sum(rels.values(), []))

    for source_1, pairs_1 in pairs_by_source.items():
      for source_2, pairs_2 in pairs_by_source.items():
        overlap_percentage_dict[source_1][source_2] = len(
            pairs_1.intersection(pairs_2))/float(len(pairs_1))
        print(source_1, source_2, overlap_percentage_dict[source_1][source_2])

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
  #config_line = "config_msh_mth	CHD|RN	None	None	None	None"
  for path, name in umls_lib.SEMANTIC_TYPE_NAMES.items(): 
    config_line = "".join(["config_msh_mth	CHD|RN	None	None	None	",
       path])
    conn = sqlite3.connect(umls_lib.DB_FILE)
    c = conn.cursor()

    pairs = umls_lib.read_subgraph_from_config_line(config_line).get_pairs(conn)
    g = MultiGraph(pairs)
    hypernym_percents_dict = g.hypernym_breakdowns(name)
  pass

if __name__ == "__main__":
  main()
