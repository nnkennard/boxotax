import sys
import collections

DUMMY_LEAF_WEIGHT = 1.0
ROOT_STR = "!!ROOT"

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

def get_transitive_reduction_from_files(input_files, root):
  edges = collections.defaultdict(list)
  negative_edges = collections.defaultdict(list)

  for input_file in input_files:
    with open(input_file, 'r') as f:
      for line in f:
        hyper, hypo, tf = line.split()
        if tf == 'true':
          edges[hyper].append(hypo)
        else:
          assert tf == 'false'
          negative_edges[hyper].append(hypo)

  hyper_nodes = set(edges.keys())
  hypo_nodes = set(sum(edges.values(),[]))
  leaves = hypo_nodes - hyper_nodes

  # Find transitive reduction of info tree only
  intransitive_edges = transitive_reduction(root, edges)

  return intransitive_edges, negative_edges, leaves


def assign_probabilities(input_file, info_files, vocab, prefixed_root):
  # Calculate transitive reduction
  (info_intransitive_edges, info_negative_edges,
      info_leaves) = get_transitive_reduction_from_files(
      info_files, prefixed_root)

  # Calculate unary weights
  unary_weights = {}
  for leaf in info_leaves:
    unary_weights[leaf] = DUMMY_LEAF_WEIGHT

  total_weight = assign_weights(prefixed_root, unary_weights,
      info_intransitive_edges)

  for node, weight in unary_weights.items():
    unary_weights[node] = weight / total_weight

  # Calculate conditional probabilities

  conditional_probabilities = collections.defaultdict(dict)
  assign_conditional_probabilities(prefixed_root,
      conditional_probabilities, info_intransitive_edges, unary_weights)

  # Print out only probabilities that are in input tree.
  (input_intransitive_edges, input_negative_edges,
      _) = get_transitive_reduction_from_files([input_file], prefixed_root)

  input_conditional_probabilities = collections.defaultdict(dict)
  assign_conditional_probabilities(prefixed_root,
      input_conditional_probabilities, input_intransitive_edges, unary_weights)

  for hyper, hypos in input_negative_edges.items():
    for hypo in hypos:
      input_conditional_probabilities[hypo][hyper] = 0.0

  input_nodes = set(input_intransitive_edges.keys())
  input_nodes.update(set.union(*input_intransitive_edges.values()))

  #with open(input_file + ".unary", 'w') as f:
  #  for node in vocab:
  #    prob = unary_weights.get(node, 0.0)
  #    f.write("\t".join([str(vocab.index(node)), str(prob)]) + "\n")

  with open(input_file + ".conditional", 'w') as f:
   for a, b_probs in input_conditional_probabilities.items():
    for b, conditional_prob in b_probs.items():
      f.write("\t".join([str(vocab.index(a)), str(vocab.index(b)),
        str(conditional_prob)]) + "\n")


def main():
  train_file, vocab_file, prefix = sys.argv[1:4]
  dev_file = train_file.replace(".train", ".dev")
  test_file = train_file.replace(".train", ".test")

  vocab = []
  with open(vocab_file, 'r') as f:
    for line in f:
      word = line.strip()
      vocab.append(word)

  prefixed_root = prefix + "_" + ROOT_STR

  assign_probabilities(train_file, [train_file], vocab, prefixed_root)
  assign_probabilities(dev_file, [train_file, dev_file], vocab, prefixed_root)
  assign_probabilities(test_file, [train_file, dev_file, test_file], vocab,
      prefixed_root)


if __name__ == "__main__":
  main()
