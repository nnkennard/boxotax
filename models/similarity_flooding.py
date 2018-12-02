"""Implementation of Similarity Flooding Algorithm.

TODO: Describe more stuff
"""

import base_sim
import collections
import fake_graphs

class NodePair(object):
  def __init__(self, a_label, b_label):
    self.a_label = a_label
    self.b_label = b_label
    self.children = []


def construct_sim_prop_graph(a_graph, a_nodes, b_graph, b_nodes):
  """Construct similarity propagation graph.

  Every pair of (node_a, node_b) gets a node in the similarity propagation
  graph.

  For every edge between (parent_a, child_a), an edge is added between
  (parent_a, x_b) and (child_a, y_b) where x_b and y_b are nodes in graph B and
  x is y's parent. The reverse edges are added as well.
  """

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

  node_pair_weights = collections.defaultdict(float)
  for node_pair in node_pairs:
    node_pair_weights[node_pair] = base_sim.phrase_edit_distance(*node_pair)
    print(node_pair_weights[node_pair])
  return node_pair_graph, node_pair_weights

def normalize(weights):
  total_weight = sum(weights.values())
  return {key:value/total_weight for key, value in weights.items()}

def run_iteration(prop_graph, prop_graph_weights):
  weight_accumulator = collections.defaultdict(float)
  for node, children in prop_graph.items():
    for child in children:
      weight_accumulator[child] += prop_graph_weights[node]
  return normalize(weight_accumulator)

def compute_fixpoint(graph, graph_weights):
  weight_map = collections.defaultdict(float)
  for _ in range(20):
    new_weights = run_iteration(graph, graph_weights)
    print(new_weights)

def main():
  graph, graph_weights = construct_sim_prop_graph(
      fake_graphs.graph_A, fake_graphs.graph_A_nodes, fake_graphs.graph_B,
      fake_graphs.graph_B_nodes)
  compute_fixpoint(graph, graph_weights)

if __name__ == "__main__":
  main()
