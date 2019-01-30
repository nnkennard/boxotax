import sys

def get_vocab_from_file(vocab_file):
  vocab = {}
  with open(vocab_file, 'r') as f:
    for line in f:
      source_id, name = line.strip().split("\t")
      vocab[source_id] = name
  return vocab

def main():
  train_file, vocab_path, source_1, source_2 = sys.argv[1:]
  assert train_file.endswith(".tsv")
  
  vocab_1 = get_vocab_from_file(vocab_path + source_1 + ".names")
  vocab_2 = get_vocab_from_file(vocab_path + source_2 + ".names")

  out_file = train_file.replace(".tsv", ".pairs")
  with open(out_file, 'w') as f_out:
    with open(train_file, 'r') as f:
      for line in f:
        parent, child, val = line.strip().split("\t")[:3]
        print(parent, child)
        f_out.write("\t".join([vocab_1[parent], vocab_2[child], val]) + "\n")

if __name__ == "__main__":
  main()
