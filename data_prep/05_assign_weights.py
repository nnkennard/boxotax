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



def get_transitive_reduction_from_files(input_files):
  graph = collections.defaultdict(list)
  negative_edges = []

  for input_file in input_files:
    with open(input_file, 'r') as f:
      for line in f:
        fields = line.strip().split("\t")
        if len(fields) == 3:
          parent, child, edge_type = fields
          if edge_type == POSITIVE:
            graph[parent].append(child)
          else:
            assert edge_type == NEGATIVE
            negative_edges.append((parent, child))
        else:
          assert len(fields) == 2
          parent, child = fields
          graph[parent].append(child)

  hyper_nodes = set(graph.keys())
  hypo_nodes = set(sum(graph.values(),[]))
  leaves = hypo_nodes - hyper_nodes

  # Find transitive reduction of info tree only
  intransitive_edges = transitive_reduction(graph)

  return intransitive_edges, negative_edges, leaves

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



def assign_probabilities(input_file, info_files):
  # Calculate transitive reduction
  (info_intransitive_edges, info_negative_edges,
      info_leaves) = get_transitive_reduction_from_files(
      info_files)

  # Calculate unary weights
  unary_weights = {}
  for leaf in info_leaves:
    unary_weights[leaf] = DUMMY_LEAF_WEIGHT

  total_weight = assign_weights(box_lib.ROOT_IDX_STR, unary_weights,
      info_intransitive_edges)

  for node, weight in unary_weights.items():
    unary_weights[node] = weight / total_weight

  # Calculate conditional probabilities

  conditional_probabilities = collections.defaultdict(dict)
  assign_conditional_probabilities(box_lib.ROOT_IDX_STR,
      conditional_probabilities, info_intransitive_edges, unary_weights)

  # Print out only probabilities that are in input tree.
  (input_intransitive_edges, input_negative_edges,
      _) = get_transitive_reduction_from_files([input_file])

  input_conditional_probabilities = collections.defaultdict(dict)
  assign_conditional_probabilities(box_lib.ROOT_IDX_STR,
      input_conditional_probabilities, input_intransitive_edges, unary_weights)

  for parent, child in input_negative_edges:
    input_conditional_probabilities[parent][child] = 0.0
    input_conditional_probabilities[child][parent] = 0.0

  input_nodes = set(input_intransitive_edges.keys())
  input_nodes.update(set.union(*input_intransitive_edges.values()))

  output_file = input_file.replace(".binary","") + ".conditional"
  with open(output_file, 'w') as f:
   for parent, child_cond_probs in input_conditional_probabilities.items():
    for child, cond_prob in child_cond_probs.items():
      rev_prob = input_conditional_probabilities[child][parent]
      f.write("\t".join([parent, child, str(cond_prob), str(rev_prob)]) + "\n")


def main():
  train_file = sys.argv[1]
  assert train_file.endswith("pairs.train.binary")
  dev_file = train_file.replace(".train.binary", ".dev")
  test_file = train_file.replace(".train.binary", ".test")

  graph = collections.defaultdict(list)

  assign_probabilities(train_file, [train_file])
  assign_probabilities(dev_file, [train_file, dev_file])
  assign_probabilities(test_file, [train_file, dev_file, test_file])




if __name__ == "__main__":
  main()
