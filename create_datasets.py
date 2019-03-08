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

def rename(filename, new_type):
  return filename.replace(".train", "." + new_type)

class DatasetPair(object):
  def __init__(self, main_file, aux_file, alignment_file):
    main_graph = Graph(main_file)
    aux_graph = Graph(aux_file)
    if len(main_graph.nodes) < len(aux_graph.nodes):
      main_graph, aux_graph = aux_graph, main_graph
    self.main_graph = main_graph
    self.aux_graph = aux_graph
    (self.vocab, self.main_indices,
        self.aux_indices) = self.get_combined_vocab()
    self.rev_vocab = collections.defaultdict(list)
    for key, value in self.vocab.items():
      self.rev_vocab[value].append(key)
    self.dev_files = (rename(main_file, "dev"), rename(aux_file, "dev"))
    self.test_files = (rename(main_file, "test"), rename(aux_file, "test"))
    (self.aligned_pairs,
        self.inter_negatives) = self.get_alignment_from_file(
            alignment_file)
    print(len(self.aligned_pairs))
    print(len(self.inter_negatives))

  def add_to_vocab(self, term):
    idx = max(self.vocab.keys()) + 1
    self.vocab[idx] = term
    self.rev_vocab[term].append(idx)
    return idx

  def get_idx_or_add(self, term):
    maybe_idx = self.rev_vocab.get(term)
    if not maybe_idx:
      #return self.add_to_vocab(term)
      return None
    else:
      return maybe_idx[0]

  def get_combined_vocab(self):
    vocab = {}
    offset = find_offset(len(self.main_graph.nodes))
    aux_indices = self.aux_graph.add_to_vocab(vocab, offset=0)
    main_indices = self.main_graph.add_to_vocab(vocab, offset=offset)
    return vocab, main_indices, aux_indices


  def get_alignment_from_file(self, alignment_file):
    aligned_pairs = []
    negative_pairs = []
    with open(alignment_file, 'r') as f:
      for line in f:
        word_1, word_2, sim = line.strip().split("\t")

        main_idx = self.get_idx_or_add(word_1)
        aux_idx = self.get_idx_or_add(word_2)
        if main_idx is None or aux_idx is None:
          continue

        if float(sim) > 0.9:
          aligned_pairs.append((main_idx, aux_idx))
        elif float(sim) < 0.1:
          # Maybe sample 0 similarity ones for this?
          negative_pairs.append((main_idx, aux_idx))
    print("num negative pairs")
    print(len(negative_pairs))
    return aligned_pairs, negative_pairs

  def print_intra_similarities(self, handle, graph):
    print("intra similarities")
    i = 0
    for hyper, hypos in graph.graph.items():
      for hypo in hypos:
        i += 1
        if i % 100 == 0:
          print(i)
        prob = graph.probs[hypo][hyper]
        self.print_line_from_names(handle, hypo, hyper, prob, 1.0)

  def print_probabilities(self, handle):
    # Print main probabilities
    self.print_intra_similarities(handle, self.main_graph)
    self.print_intra_similarities(handle, self.aux_graph)
    for n1, n2 in self.inter_negatives:
      self.print_line(handle, n1, n2, 0.0, 0.0)
    for n1, n2 in self.aligned_pairs:
      self.print_line(handle, n1, n2, 1.0, 1.0)

  def print_vocab(self, handle):
    for idx, label in self.vocab.items():
      handle.write("\t".join([str(idx), label]) + "\n")

  def print_dataset(self, filename):
    (train_file, dev_file, test_file, vocab_file) = get_filenames(filename)
    print(train_file)
    with open(train_file, 'w') as f:
      self.print_probabilities(f)
    with open(dev_file, 'w') as f:
      for old_dev_file in self.dev_files:
        with open(old_dev_file, 'r') as g:
          print(old_dev_file)
          for line in g:
            hypo, hyper, p1 = line.strip().split("\t")
            self.print_line_from_names(f, hypo, hyper, p1, 1.0)
    with open(test_file, 'w') as f:
      for old_test_file in self.test_files:
        print(old_test_file)
        with open(old_test_file, 'r') as g:
          for line in g:
            hypo, hyper, p1 = line.strip().split("\t")
            self.print_line_from_names(f, hypo, hyper, p1, 1.0)
    with open(vocab_file, 'w') as f:
      self.print_vocab(f)


  def print_line_from_names(self, handle, hypo, hyper, prob1, prob2):
    hypo_idx = self.get_idx_or_add(hypo) 
    hyper_idx = self.get_idx_or_add(hyper) 
    if hypo_idx is None or hyper_idx is None:
      return
    self.print_line(handle, hypo_idx, hyper_idx, prob1, prob2)

  def print_line(self, handle, hypo, hyper, prob1, prob2):
    # TODO: Check if there is anything weird here with two names being the same
    # in different datasets.
    handle.write("\t".join([str(i) for i in [hypo, hyper, prob1, prob2]]) +  "\n")


def get_filenames(root_filename):
  return (root_filename + ".train",
      root_filename + ".dev",
      root_filename + ".test",
      root_filename + ".vocab")


def make_root_name(train_dir, name1, name2):
  return train_dir + "/" + "-".join(sorted([name1, name2]))


def main():
  train_dir, name_1, name_2 = sys.argv[1:4]

  file_1 = "".join([train_dir, "/", name_1, ".train"])
  file_2 = "".join([train_dir, "/", name_2, ".train"])
  
  sim_file_name = train_dir + "/" + "-".join(sorted([name_1, name_2])) + ".sim"

  k = DatasetPair(file_1, file_2, sim_file_name)
  k.print_dataset(make_root_name(train_dir, name_1, name_2))

if __name__ == "__main__":
  main()
