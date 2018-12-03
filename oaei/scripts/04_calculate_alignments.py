import oaei_lib
import subprocess
import sys

def main():
  data_path = sys.argv[1]

  #for pair, datasets in oaei_lib.PAIR_TO_DATASET_NAMES.items():
  for pair, datasets in [("fma2nci", ["fma", "nci"])]:
    for dataset in datasets:
      print("Calculating affinities for " + pair + " " + dataset)
      vocab_path = "".join([data_path, '/small/', pair, '/', pair,
          '.vocab'])
      subprocess.call(["python", "04a_base_sim.py", vocab_path])

if __name__ == "__main__":
  main()
