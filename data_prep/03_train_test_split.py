import sys
import random

def main():
  input_file = sys.argv[1]
  random.seed(78)
  train_frac, dev_frac, test_frac = [0.8, 0.1, 0.1]

  input_lines = []
  with open(input_file, 'r') as f:
    input_lines = f.readlines()
  random.shuffle(input_lines)

  num_examples = float(len(input_lines))
  train_end = int(train_frac * num_examples)
  dev_end = int((train_frac + dev_frac) * num_examples)

  output_suffixes = ["train.no_neg", "dev", "test"]

  train_examples = input_lines[:train_end]
  dev_examples = input_lines[train_end:dev_end]
  test_examples = input_lines[dev_end:]

  for suffix, dataset in zip(output_suffixes, [train_examples, dev_examples,
    test_examples]):
    suffixed_filename = input_file + "." + suffix
    with open(suffixed_filename, 'w') as f:
      f.write("".join(dataset))

if __name__ == "__main__":
  main()
