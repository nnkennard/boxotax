import random
import sys

import base_sim

def get_vocab(train_filename):
  vocab = {}
  vocab_filename = train_filename.replace(".train",".vocab")
  with open(vocab_filename, 'r') as f:
    for line in f:
      print(line)
      idx, name = line.strip().split("\t")
      vocab[idx] = name

  return vocab

def get_nodes(train_filename):
  nodes = set()
  with open(train_filename, 'r') as f:
    for line in f:
      nodes.update(line.strip().split("\t"))
  return list(nodes)

def main():
  train_file_1, train_file_2 = sys.argv[1:3]
  assert train_file_1.endswith(".train")
  assert train_file_2.endswith(".train")
  vocab_1 = get_vocab(train_file_1)
  vocab_2 = get_vocab(train_file_2)
  for k, v in vocab_2.items():
    print(k,v)
    if "000" in k:
      break

  nodes_1 = get_nodes(train_file_1)
  nodes_2 = get_nodes(train_file_2)

  negative_ratio = 0.05

  num_negatives = int(negative_ratio * float(min(len(nodes_1), len(nodes_2))))
  num_negatives = 10

  negative_edges = []

  while len(negative_edges) < num_negatives:
    label_1 = vocab_1[random.choice(nodes_1)]
    label_2 = vocab_2[random.choice(nodes_2)]
    sim = base_sim(label_1, label_2)
    print(label_1, label_2, sim)



if __name__ == "__main__":
  main()
