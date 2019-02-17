import collections

import box_data_lib

ROOT_STR = "!!ROOT"
ROOT_IDX_STR = str(0).zfill(6)

NEGATIVE_RATIO = 1.0

def translate_to_interim_vocab(pair_file):
  vocab = set([ROOT_STR])
  pairs = []
  with open(pair_file, 'r') as f:
    for line in f:
      hypo, hyper = line.strip().split()
      vocab.update([hypo, hyper])
      pairs.append((hypo, hyper))
  assert len(vocab) < 999999
  vocab_map = {word:str(idx).zfill(6) for word in sorted(list(vocab))}
  translated_pairs = [(vocab_map(x), vocab_map(y)) for x, y in pairs]
  interim_vocab = {v: k for k, v in vocab_map.iteritems()}
  return interim_vocab, translated_pairs

def pairs_to_graph(pairs):
  # Check for root
  graph = collections.defaultdict(list)
  for parent, child in pairs:
    graph[parent].append(child)
  return graph

def get_transitive_closure(graph, root, ancestor_list, seen, edges):
  if root in seen:
    return
  seen.append(root)
  for child in sorted(graph[root]):
    get_transitive_closure(graph, child, ancestor_list+[root], seen, edges)
  for ancestor in ancestor_list:
    edges.append((ancestor, root))


def get_transitive_closure(pairs):
  graph = pairs_to_graph(pairs)

  transitive_edges = []
  box_lib.get_transitive_closure(
      graph, box_lib.ROOT_IDX_STR, [], [], transitive_edges)

  return transitive_edges

class SingleSourceDataset(object):
  def __init__(self, transitive_pairs):
    self.transitive_pairs = transitive_pairs
    self.train, self.dev, self.test = self.make_train_test_split()
    self.negatives = self.calculate_negatives()
    pass

  def make_train_test_split(self):
    random.seed(78)
    train_frac, dev_frac, test_frac = [0.8, 0.1, 0.1]

    self.transitive_pairs = sorted(self.transitive_pairs)
    random.shuffle(self.transitive_pairs)

    num_examples = float(len(self.transitive_pairs))
    train_end = int(train_frac * num_examples)
    dev_end = int((train_frac + dev_frac) * num_examples)

    train_examples = list(self.transitive_pairs[:train_end])
    dev_examples = list(self.transitive_pairs[train_end:dev_end])
    test_examples = list(self.transitive_pairs[dev_end:])

    return train_examples, dev_examples, test_examples

  def calculate_negatives():
    intransitive_train = self.get_intransitive_train()
    self.negatives = []
    # Add the negatives

def main():
  interim_vocab, translated_pairs = translate_to_interim_vocab(pair_file)
  transitive_pairs = get_transitive_closure(translated_pairs)
  single_source_dataset = SingleSourceDataset(transitive_pairs)

if __name__ == "__main__":
  main()
