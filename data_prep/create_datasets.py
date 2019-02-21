import collections
import time
import tqdm

def find_offset(num_nodes):
  num_digits = len(str(num_nodes))
  return 10 ** num_digits

class Graph(object):
  def __init__(self, pairs):
    self.graph, self.nodes = self.graph_from_pairs(pairs)

  def graph_from_pairs(self, pair_list):
    graph = collections.defaultdict(list)
    nodes = set()
    for hypo, hyper in pair_list:
      graph[hyper].append(hypo)
      nodes.update([hypo, hyper])
    return graph, nodes

  def add_to_vocab(offset):
    for i, node in enumerate(sorted(self.nodes)):
      if i == 0:
        assert node == "!!ROOT"
        vocab[0] = node
      else:
        vocab[offset + i] = node


class SimilarityFunction(object):
  def __init__(self, func, align_threshold, unrelated_threshold):
    self.func = func
    self.align_threshold = align_threshold
    self.unrelated_threshold = unrelated_threshold


class DatasetPair(object):
  def __init__(self, main_pairs, aux_pairs, similarity_fn, alignment=None):
    main_graph = Graph(main_pairs)
    aux_graph = Graph(aux_pairs)
    if len(main_graph.nodes) < len(aux_graph.nodes):
      main_graph, aux_graph = aux_graph, main_graph
    self.main_graph = main_graph
    self.aux_graph = aux_graph
    self.vocab = self.get_vocab()

    if alignment is not None:
      self.aligned_pairs, self.inter_negatives = self.get_alignment_from_file()
    else:
      self.aligned_pairs, self.inter_negatives =
      self.get_alignment_from_similarity(similarity_func)


  def get_vocab(self):
    vocab = {}
    offset = find_offset(len(main_graph.nodes))
    self.aux_graph.add_to_vocab(vocab, offset=0)
    self.main_graph.add_to_vocab(vocab, offset=offset)


# Delete this
class DatasetPair(object):
  def __init__(self, main_dataset, aux_dataset, similarity_func,
      alignment_info=None):
    #self.main_dataset = main_dataset
    #self.aux_dataset = aux_dataset

    self.main_graph = Graph(main_dataset)
    self.aux_graph = Graph(aux_dataset)
  def get_aligned_pairs(self, similarity_func):
    aligned_pairs = []
    negative_pairs = []
    for main_node in main_dataset.nodes:
      for aux_node in aux_dataset.nodes:
        sim = similarity_func.func(main_node.label, aux_node.label)
        if sim > similarity_func.align_threshold:
          aligned_pairs.append((main_node, aux_node))
        elif sim < similarity_func.unrelated_threshold:
          negative_pairs.append((main_node, aux_node))


    pass

  def get_inter_negatives(self):

def main():
  pass

if __name__ == "__main__":
  main()
