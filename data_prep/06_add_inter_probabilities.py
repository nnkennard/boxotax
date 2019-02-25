import random
import sys

import base_sim

def get_vocab(vocab_filename):
  vocab = {}
  with open(vocab_filename, 'r') as f:
    for line in f:
      idx, name = line.strip().split("\t")
      vocab[idx] = name

  return vocab

def get_nodes(train_filename):
  nodes = set()
  with open(train_filename, 'r') as f:
    for line in f:
      nodes.update(line.strip().split("\t"))
  return list(nodes)

def get_nodes_and_vocab(train_filename):
  nodes = get_nodes(train_filename)
  vocab = get_vocab(train_filename.replace("tpairs.train.no_neg", "vocab"))
  return nodes, vocab

def main():
  random.seed(43)
  train_file_1, train_file_2, output_file = sys.argv[1:4]
  assert train_file_1.endswith(".train.no_neg")
  assert train_file_2.endswith(".train.no_neg")


  nodes_1, vocab_1 = get_nodes_and_vocab(train_file_1)
  nodes_2, vocab_2 = get_nodes_and_vocab(train_file_2)

  negative_ratio = 1.0

  num_negatives = int(negative_ratio * float(min(len(nodes_1), len(nodes_2))))

  negative_edges = []
  alignment_edges = []

  while len(negative_edges) < num_negatives:
    if len(negative_edges) % 1000 == 0:
      print(str(len(negative_edges)) + "/" + str(num_negatives))
    node_1 = random.choice(nodes_1)
    node_2 = random.choice(nodes_2)
    label_1 = vocab_1[node_1]
    label_2 = vocab_2[node_2]
    sim = base_sim.base_sim(label_1, label_2)
    if sim < 0.5:
      negative_edges.append((node_1, node_2))

  with open(output_file, 'w') as f:
    for parent, child in negative_edges:
      f.write("\t".join([parent, child, "0.0", "0.0"]) + "\n")


if __name__ == "__main__":
  main()
