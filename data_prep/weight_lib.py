import collections
import sys

import box_lib

POSITIVE = "1"
NEGATIVE = "0"
DUMMY_LEAF_WEIGHT = 1.0


def assign_weights(root, unary_weights, edges):
  """Assign weights, propagating up an intransitive graph."""
  if root not in unary_weights:
    total_prob = 0.0
    for child in edges[root]:
      total_prob += assign_weights(child, unary_weights, edges)
    unary_weights[root] = total_prob

  return unary_weights[root]


def bfs_order(root, edges):
  """BFS over tree given a root.
    Args:
      root:
      edges: edges in edge list format
  """
  bfs = []
  to_visit = [root]
  while len(to_visit):
    current = to_visit.pop(0)
    bfs.append(current)
    to_visit+=[child for child in edges[current]
        if child not in bfs
        and child not in to_visit]
  return bfs


def transitive_reduction(edges):
  """Returns the transitive reduction of a edges."""
  all_nodes = set(edges.keys())
  all_nodes.update(sum(edges.values(), []))
  intransitive_edges_constructor = collections.defaultdict(set)
  for parent, children in edges.items():
    intransitive_edges_constructor[parent].update(children)


  for i in bfs_order(box_lib.ROOT_IDX_STR, edges):
    for j in edges[i]:
      for k in edges[j]:
        if k in edges[i]:
          intransitive_edges_constructor[i] -= set([k])

  return intransitive_edges_constructor


def get_transitive_reduction_from_pairs(input_pairs):
  graph = collections.defaultdict(list)
  negative_edges = []

  for pair_list in input_pairs:
    for hypo, hyper in pair_list:
      graph[hyper].append(hypo)

  hyper_nodes = set(graph.keys())
  hypo_nodes = set(sum(graph.values(),[]))
  leaves = hypo_nodes - hyper_nodes
  orphans = hyper_nodes - hypo_nodes - set([box_lib.ROOT_IDX_STR])
  for orphan in orphans:
    graph[box_lib.ROOT_IDX_STR].append(orphan)

  # Find transitive reduction of info tree only
  intransitive_edges = transitive_reduction(graph)

  return intransitive_edges, leaves


def assign_conditional_probabilities(parent, conditional_probabilities,
    edges, unary_weights):
  """Assign conditional probabilities."""
  if parent in edges:
    for child in edges[parent]:
      # P(child|parent) = P(child)/P(parent)
      conditional_probabilities[parent][child] = unary_weights[
          child]/unary_weights[parent]
      # P(parent|child) = 1.0
      conditional_probabilities[child][parent] = 1.0

      assign_conditional_probabilities(child, conditional_probabilities,
          edges, unary_weights)


def get_unary_weights(info_pairs):
  # Calculate transitive reduction
  (info_intransitive_edges, info_leaves) = get_transitive_reduction_from_pairs(
      info_pairs)

  # Calculate unary weights
  unary_weights = {}
  for leaf in info_leaves:
    unary_weights[leaf] = DUMMY_LEAF_WEIGHT


  total_weight = assign_weights(box_lib.ROOT_IDX_STR, unary_weights,
      info_intransitive_edges)

  for node, weight in unary_weights.items():
    unary_weights[node] = weight / total_weight

  return unary_weights


def assign_conditional_probabilities_wrapper(intransitive_edges, weights):
  conditional_probabilities = collections.defaultdict(dict)
  assign_conditional_probabilities(box_lib.ROOT_IDX_STR,
      conditional_probabilities, intransitive_edges, weights)
  return conditional_probabilities


def assign_probabilities(input_pairs, info_pairs):

  unary_weights = get_unary_weights(info_pairs)

  input_conditional_probabilities = collections.defaultdict(dict)
  for hypo, hyper in input_pairs:
    cond_prob = unary_weights[hypo]/unary_weights[hyper]
    input_conditional_probabilities[hypo][hyper] = cond_prob
    input_conditional_probabilities[hyper][hypo] = 1.0

  return input_conditional_probabilities

def assign_all_probabilities(train_pairs, dev_pairs, test_pairs):
  train_probs = assign_probabilities(train_pairs, [train_pairs])
  dev_probs = assign_probabilities(dev_pairs, [train_pairs, dev_pairs])
  test_probs = assign_probabilities(test_pairs, [train_pairs, dev_pairs, test_pairs])
  return train_probs, dev_probs, test_probs
