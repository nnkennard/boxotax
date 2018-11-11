import subprocess
import sys

PAIR_TO_DATASET_NAMES = {
    "fma2nci": ["fma", "nci"],
    "fma2snomed": ["fma", "snomed"],
    "snomed2nci": ["nci", "snomed"]
    }

def main():
  data_path = sys.argv[1]

  for pair, datasets in PAIR_TO_DATASET_NAMES.items():
    for dataset in datasets:
      print("".join([
        "python streamline_data.py ", data_path,'/small/', pair,
        "/", dataset, ".owl ", dataset]))
      print("".join([
        "python split_test_train.py ", data_path, '/small/', pair,
        "/", dataset, ".out ", dataset]))


if __name__ == "__main__":
  main()
