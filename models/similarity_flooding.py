"""Implementation of Similarity Flooding Algorithm."""

import collections
import base_sim



class NodePair(object):
  def __init__(self, a_label, b_label):
    self.a_label = a_label
    self.b_label = b_label
    self.children = []


  

graph_A = {
    'abcd': ['abcd1', 'abcd2'],
    'abcd1': ['abcd2'],
    'abcd2': []
    }

graph_A_nodes = ['abcd', 'abcd1', 'abcd2']

graph_B = {
    'bcde': ['bcde1', 'bcde2'],
    'bcde2': ['bcde1'],
    'bcde1': []
    }

graph_B_nodes = ['bcde', 'bcde1', 'bcde2']

def construct_sim_prop_graph(a_graph, a_nodes, b_graph, b_nodes):
  node_pairs = set()
  node_pair_graph = collections.defaultdict(list)
  for a_node in a_nodes:
    for b_node in b_nodes:
      node_pairs.add((a_node, b_node))
      for a_child in a_graph[a_node]:
        for b_child in b_graph[b_node]:
          node_pair_graph[(a_node, b_node)].append((a_child, b_child))
          node_pair_graph[(a_child, b_child)].append((a_node, b_node))
          node_pairs.add((a_child, b_child))

  print(node_pairs)
  print(node_pair_graph)

  node_pair_weights = collections.defaultdict(float)
  for node_pair in node_pairs:
    node_pair_weights[node_pair] = base_sim.phrase_edit_distance(*node_pair)
    print(node_pair_weights[node_pair])
  return node_pair_graph

def compute_fixpoint():
  weight_map = collections.defaultdict(float)
  # Assign initial similarity scores
  # Propagate
  pass

def main():
  construct_sim_prop_graph(graph_A, graph_A_nodes, graph_B, graph_B_nodes)
  pass

if __name__ == "__main__":
  main()
