import collections
import sys
import time
import tqdm

def find_offset(num_nodes):
  num_digits = len(str(num_nodes))
  return 10 ** num_digits

class Graph(object):
  def __init__(self, file_name):
    self.graph, self.probs, self.nodes = self.graph_from_file(file_name)

  def graph_from_file(self, file_name):
    graph = collections.defaultdict(list)
    probs = collections.defaultdict(dict)
    nodes = set()

    with open(file_name, 'r') as f:
      for line in f:
        hypo, hyper, prob = line.strip().split("\t")
        graph[hyper].append(hypo)
        probs[hypo][hyper] = prob
        nodes.update([hypo, hyper])
    return graph, probs, sorted(list(nodes))

  def add_to_vocab(self, vocab, offset):
    indices = [0]
    for i, node in enumerate(sorted(self.nodes)):
      if i == 0:
        assert node == "!!ROOT"
        vocab[0] = node
      else:
        index = offset + i
        vocab[index] = node
        indices.append(index)
    return sorted(indices)


class SimilarityFunction(object):
  def __init__(self, func, align_threshold, unrelated_threshold):
    self.func = func
    self.align_threshold = align_threshold
    self.unrelated_threshold = unrelated_threshold


class DatasetPair(object):
  def __init__(self, main_file, aux_file, similarity_fn, alignment=None):
    main_graph = Graph(main_file)
    aux_graph = Graph(aux_file)
    if len(main_graph.nodes) < len(aux_graph.nodes):
      main_graph, aux_graph = aux_graph, main_graph
    self.main_graph = main_graph
    self.aux_graph = aux_graph
    (self.vocab, self.main_indices,
        self.aux_indices) = self.get_combined_vocab()

    if alignment is not None:
      self.aligned_pairs, self.inter_negatives = self.get_alignment_from_file()
    else:
      (self.aligned_pairs,
          self.inter_negatives) = self.get_alignment_from_similarity(
              similarity_func)

  def get_combined_vocab(self):
    vocab = {}
    offset = find_offset(len(self.main_graph.nodes))
    aux_indices = self.aux_graph.add_to_vocab(vocab, offset=0)
    main_indices = self.main_graph.add_to_vocab(vocab, offset=offset)
    return vocab, main_indices, aux_indices

  def get_alignment_from_similarity(self, similarity_func):
    aligned_pairs = []
    negative_pairs = []
    for main_idx in self.main_indices:
      for aux_idx in self.aux_indices:
        sim = similarity_func.func(vocab[main_idx], vocab[aux_idx])
        if sim > similarity_func.align_threshold:
          aligned_pairs.append((main_idx, aux_idx))
        elif sim < similarity_func.unrelated_threshold:
          negative_pairs.append((main_idx, aux_idx))

    return aligned_pairs, negative_pairs

  def print_probabilities():
    # Print main probabilities
    for hyper, hypos in main_graph.graph:
      for hypo in hypos:
        prob = main_graph.probs[hypo][hyper]
        print_line(handle, hypo, hyper, prob, 1.0)
    # Print aux probabilities
    for hyper, hypos in aux_graph.graph:
      for hypo in hypos:
        prob = main_graph.probs[hypo][hyper]
        print_line(handle, hypo, hyper, prob, 1.0)

    for n1, n2 in self.inter_negatives:
      print_line(handle, n1, n2, 0.0, 0.0)
    for n1, n2 in self.alignments:
      print_line(handle, n1, n2, 1.0, 1.0)

  def print_vocab():
    pass

  def print_dataset(filename):
    pass

def main():
 file_1, file_2 = sys.argv[1:3]
 k = DatasetPair(file_1, file_2, None, None)

if __name__ == "__main__":
  main()
