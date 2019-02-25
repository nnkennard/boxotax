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
      assert similarity_fn is None
      self.aligned_pairs, self.inter_negatives = self.get_alignment_from_file()
    else:
      (self.aligned_pairs,
          self.inter_negatives) = self.get_alignment_from_similarity(
              similarity_fn)

  def get_combined_vocab(self):
    vocab = {}
    offset = find_offset(len(self.main_graph.nodes))
    aux_indices = self.aux_graph.add_to_vocab(vocab, offset=0)
    main_indices = self.main_graph.add_to_vocab(vocab, offset=offset)
    return vocab, main_indices, aux_indices

  def get_alignment_from_similarity(self, similarity_func):
    aligned_pairs = []
    negative_pairs = []
    pbar = tqdm.tqdm(total=len(self.main_indices))
    for main_idx in self.main_indices:
      for aux_idx in self.aux_indices:
        sim = similarity_func.func(self.vocab[main_idx], self.vocab[aux_idx])
        if sim > similarity_func.align_threshold:
          aligned_pairs.append((main_idx, aux_idx))
        elif sim < similarity_func.unrelated_threshold:
          negative_pairs.append((main_idx, aux_idx))
      pbar.update(1)
    pbar.close()

    return aligned_pairs, negative_pairs

  def get_alignment_from_file(self, alignment_file):
    aligned_pairs = []
    negative_pairs = []
    with open(alignment_file, 'r') as f:
      for line in f:
        word1, word2, similarity = line.strip().split()
        if float(sim) > 0.9:
          aligned_pairs.append((main_idx, aux_idx))
        elif float(sim) < 0.1:
          # Maybe sample 0 similarity ones for this?
          negative_pairs.append((main_idx, aux_idx))
    return aligned_pairs, negative_pairs


  def print_line(handle, hypo, hyper, prob1, prob2):
    handle.write("\t".join([hypo, hyper, str(prob1), str(prob2)]) + "\n")

  def print_intra_similarities(handle, graph):
    for hyper, hypos in graph.graph.items():
      for hypo in hypos:
        prob = graph.probs[hypo][hyper]
        print_line(handle, hypo, hyper, prob, 1.0)

  def print_probabilities(handle):
    # Print main probabilities
    print_intra_similarities(handle, self.main_graph)
    print_intra_similarities(handle, self.aux_graph)
    for n1, n2 in self.inter_negatives:
      print_line(handle, n1, n2, 0.0, 0.0)
    for n1, n2 in self.alignments:
      print_line(handle, n1, n2, 1.0, 1.0)

  def print_vocab(handle):
    for idx, label in self.vocab.items():
      handle.write("\t".join([idx, label]) + "\n")

  def print_dataset(filename):
    with open(train_path, 'w') as f:
      print_probabilities(f)
    with open(vocab_path, 'w') as f:
      print_vocab(f)
    pass

def main():
 file_1, file_2 = sys.argv[1:3]
 if len(sys.argv) > 3:
   sim_file_name = sys.argv[3]
 else:
   sim_file_name = None

 exact_similarity_func = SimilarityFunction(
     lambda x, y: 1.0 if x == y else 0.0, 0.05, 0.95)
 if sim_file is not None: 
  k = DatasetPair(file_1, file_2, None, sim_file_name)
 else:
  k = DatasetPair(file_1, file_2, exact_similarity_func, None)
  k.print_probabilities(sys.stdout)

if __name__ == "__main__":
  main()
