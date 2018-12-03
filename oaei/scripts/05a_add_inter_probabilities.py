import oaei_lib
import subprocess
import sys

def get_vocabs(datasets):
  pass

SIMILARITY_THRESHOLD = 0.7
ALIGNMENT_PROBAILITY = "0.9"

def main():
  similarity_path, vocab_path = sys.argv[1:3]
  output_path = vocab_path.replace("vocab", "conditional")

  #for pair, datasets in oaei_lib.PAIR_TO_DATASET_NAMES.items():
  for pair, datasets in [("fma2nci", ["fma", "nci"])]:
    print("Adding inter probabilities ", pair)

    vocab = []
    with open(vocab_path, 'r') as f:
      for line in f:
        vocab.append(line.strip())
    inter_probabilities = []
    with open(similarity_path, 'r') as f:
      for line in f:
        index_1, index_2, score = line.strip().split()
        if float(score) > SIMILARITY_THRESHOLD:
          inter_probabilities.append((index_1, index_2,
            ALIGNMENT_PROBAILITY))
          inter_probabilities.append((index_2, index_1,
            ALIGNMENT_PROBAILITY))

      with open(output_path, 'w') as f_out:
        for index_1, index_2, prob in inter_probabilities:
          f_out.write("\t".join([
            str(index_1), str(index_2), prob]) + "\n")

if __name__ == "__main__":
  main()
