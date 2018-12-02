import oaei_lib
import subprocess
import sys

def main():
  data_path = sys.argv[1]

  #for pair, datasets in oaei_lib.PAIR_TO_DATASET_NAMES.items():
  for pair, datasets in [("fma2nci", ["fma", "nci"])]:
    for dataset in datasets:
      print("Assigning probabilities for " + pair + " " + dataset)
      file_path = "".join([data_path, '/small/', pair, '/', dataset,
          '.out.train'])
      subprocess.call(["python", "03a_assign_probabilities.py", file_path])

if __name__ == "__main__":
  main()
