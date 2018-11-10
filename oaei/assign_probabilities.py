import sys
import collections

DUMMY_LEAF_WEIGHT = 1.0
ROOT_INDEX = "0"

def assign_conditional_probabilities(root, conditional_probabilities,
    edges, unary_weights):
  """Assign conditional probabilities."""
  if root in edges:
    for child in edges[root]:
      # P(hypo|hyper) = P(hypo)/P(hyper)
      conditional_probabilities[child][root] =unary_weights[
          child]/unary_weights[root]
      # P(hyper|hypo) = 1.0
      conditional_probabilities[root][child] = 1.0

      assign_conditional_probabilities(child, conditional_probabilities,
          edges, unary_weights)


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

def transitive_reduction(root, edges):
  """Returns the transitive reduction of a edges."""
  all_nodes = set(edges.keys())
  all_nodes.update(sum(edges.values(), []))
  intransitive_edges_constructor = collections.defaultdict(set)
  for parent, children in edges.items():
    intransitive_edges_constructor[parent].update(children)

  for i in bfs_order(root, edges):
    for j in edges[i]:
      for k in edges[j]:
        if k in edges[i]:
          intransitive_edges_constructor[i] -= set([k])

  return intransitive_edges_constructor


def main():

  input_file = sys.argv[1]
  edges = collections.defaultdict(list)

  with open(input_file, 'r') as f:
    for line in f:
      hyper, hypo, _, _ = line.split()
      edges[hyper].append(hypo)

  hyper_nodes = set(edges.keys())
  hypo_nodes = set(sum(edges.values(),[]))
  leaves = hypo_nodes - hyper_nodes

  # Find transitive reduction
  intransitive_edges = transitive_reduction(ROOT_INDEX, edges)

  # Calculate unary weights using transitive reduction
  unary_weights = {}
  for leaf in leaves:
    unary_weights[leaf] = DUMMY_LEAF_WEIGHT

  total = assign_weights(ROOT_INDEX, unary_weights, intransitive_edges)

  # Calculate conditional probabilities using transitive closure
  conditional_probabilities = collections.defaultdict(dict)

  assign_conditional_probabilities(ROOT_INDEX,
      conditional_probabilities, edges, unary_weights)

  for a, b_probs in conditional_probabilities.items():
    for b, conditional_prob in b_probs.items():
      print(a+"\t"+b+"\t"+str(conditional_prob))


  unary_probabilities = {}
  for key, value in unary_weights.items():
    unary_probabilities[key] = value/total

if __name__ == "__main__":
  main()
