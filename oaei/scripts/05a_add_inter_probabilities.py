import oaei_lib
import random
import subprocess
import sys

def get_vocabs(datasets):
  pass

SIMILARITY_THRESHOLD = 0.9
ALIGNMENT_PROBABILITY = "1.0"
DISJOINT_PROBABILITY = "0.0"

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
    inter_pairs = []
    inter_nonpairs = []

    with open(similarity_path, 'r') as f:
      for line in f:
        index_1, index_2, score = line.strip().split()
        if float(score) > SIMILARITY_THRESHOLD:
          inter_pairs.append((index_1, index_2))
        else:
          inter_nonpairs.append((index_1, index_2))

    selected_nonpairs = random.sample(inter_nonpairs, len(inter_pairs))
    print(len(inter_pairs))
    print(len(selected_nonpairs))

    with open(output_path, 'w') as f_out:
      for index_1, index_2 in inter_pairs:
        f_out.write("\t".join([
          str(index_1), str(index_2), ALIGNMENT_PROBABILITY]) + "\n")
        f_out.write("\t".join([
          str(index_2), str(index_1), ALIGNMENT_PROBABILITY]) + "\n")
      for index_1, index_2 in selected_nonpairs:
        f_out.write("\t".join([
          str(index_1), str(index_2), DISJOINT_PROBABILITY]) + "\n")
        f_out.write("\t".join([
          str(index_2), str(index_1), DISJOINT_PROBABILITY]) + "\n")


if __name__ == "__main__":
  main()
