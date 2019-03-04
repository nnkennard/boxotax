import collections
import random
import sys
import time
import tqdm

import weight_lib

ROOT_STR = "!!ROOT"
ROOT_IDX_STR = str(0).zfill(6)

NEGATIVE_RATIO = 1.0

def translate_to_interim_vocab(pair_file):
  print("Translating to interim vocab")
  vocab = set([ROOT_STR])
  pairs = []
  with open(pair_file, 'r') as f:
    for line in f:
      hypo, hyper = line.strip().split("\t")
      vocab.update([hypo, hyper])
      pairs.append((hypo, hyper))
  assert len(vocab) < 999999
  vocab_map = {word:str(idx).zfill(6) for idx, word in
      enumerate(sorted(list(vocab)))}
  translated_pairs = [(vocab_map[x], vocab_map[y]) for x, y in pairs]
  interim_vocab = {v: k for k, v in vocab_map.items()}
  return interim_vocab, translated_pairs


def pairs_to_graph(pairs):

  graph = collections.defaultdict(list)
  for hypo, hyper in pairs:
    graph[hyper].append(hypo)

  if ROOT_IDX_STR not in graph:
    superclass_nodes = set(graph.keys())
    subclass_nodes = set(sum(graph.values(), []))
    all_nodes = superclass_nodes.union(subclass_nodes)
    non_subclass_nodes = all_nodes - subclass_nodes
    graph[ROOT_IDX_STR] = list(non_subclass_nodes)

  return graph


def get_transitive_closure(graph, root, ancestor_list, seen, edges):
  if root in seen:
    return
  seen.append(root)
  for child in sorted(graph[root]):
    get_transitive_closure(graph, child, ancestor_list + [root], seen, edges)
  for ancestor in ancestor_list:
    edges.append((root, ancestor))


def get_transitive_closure_from_pairs(pairs):
  graph = pairs_to_graph(pairs)

  transitive_edges = []
  print(ROOT_IDX_STR in graph.keys())
  print(ROOT_STR in graph.keys())
  print(list(graph.keys())[:10])
  get_transitive_closure(graph, ROOT_IDX_STR, [], [], transitive_edges)

  return transitive_edges


def write_to_file(subset, probs, file_name, vocab):
  if probs is None:
    prob = 0.0
  else:
    prob = None

  with open(file_name, 'w') as f:
    for hypo, hyper in subset:
      if prob is not 0.0:
        prob = probs[hypo][hyper]

      f.write(
          "\t".join(
            [vocab[hypo], vocab[hyper], str(prob)]) + "\n")


class SingleSourceDataset(object):
  def __init__(self, transitive_pairs, vocab):
    self.transitive_pairs = transitive_pairs
    print(len(self.transitive_pairs))
    self.vocab = vocab
    self.nodes = self.get_node_list()
    self.train, self.dev, self.test = self.make_train_test_split()
    self.negatives = self.calculate_negatives()
    (self.train_probs, self.dev_probs,
        self.test_probs) = weight_lib.assign_all_probabilities(self.train,
            self.dev, self.test)

  def get_node_list(self):
    node_set = set()
    for hypo, hyper in self.transitive_pairs:
      node_set.update([hypo, hyper])
    return sorted(list(node_set))

  def make_train_test_split(self):
    random.seed(78)
    train_frac, dev_frac, test_frac = [0.8, 0.1, 0.1]

    self.transitive_pairs = sorted(self.transitive_pairs)
    random.shuffle(self.transitive_pairs)

    num_examples = float(len(self.transitive_pairs))
    print(num_examples)
    train_end = int(train_frac * num_examples)
    dev_end = int((train_frac + dev_frac) * num_examples)

    train_examples = list(self.transitive_pairs[:train_end])
    dev_examples = list(self.transitive_pairs[train_end:dev_end])
    test_examples = list(self.transitive_pairs[dev_end:])

    return train_examples, dev_examples, test_examples

  def calculate_negatives(self):
    negative_edges = []
    total_negative_edges = NEGATIVE_RATIO * len(self.train)

    pbar = tqdm.tqdm(total=total_negative_edges)
    while len(negative_edges) < total_negative_edges:
      parent = random.choice(self.nodes)
      child = random.choice(self.nodes)
      if parent == child:
        continue
      elif (parent, child) in self.transitive_pairs:
        continue
      else:
        negative_edges.append((parent, child))
        pbar.update(1)
    pbar.close()

    return negative_edges

  def print_datasets(self, path, dataset_name):
    file_prefix = path + "/" + dataset_name
    subset_probs_suffix = [(self.train, self.train_probs, ".train"),
        (self.dev, self.dev_probs, ".dev"),
        (self.test, self.test_probs, ".test"),
        (self.negatives, None, ".neg")]
    for subset, probs, suffix in subset_probs_suffix:
      file_name = file_prefix + suffix
      write_to_file(subset, probs, file_name, self.vocab)


def main():
  pair_file, name = sys.argv[1:]
  output_dir = "/".join(pair_file.split("/")[:-1])

  interim_vocab, translated_pairs = translate_to_interim_vocab(pair_file)
  transitive_pairs = get_transitive_closure_from_pairs(translated_pairs)

  single_source_dataset = SingleSourceDataset(transitive_pairs, interim_vocab)
  single_source_dataset.print_datasets(output_dir, name)

if __name__ == "__main__":
  main()
