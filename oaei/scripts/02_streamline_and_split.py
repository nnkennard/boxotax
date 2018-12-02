import subprocess
import sys

PAIR_TO_DATASET_NAMES = {
    "fma2nci": ["fma", "nci"],
    "fma2snomed": ["fma", "snomed"],
    "snomed2nci": ["nci", "snomed"]
    }

def main():
  data_path = sys.argv[1]

  for dataset in ["fma", "snomed", "nci"]:
    file_path = "".join([data_path, '/large/', dataset, '.owl'])
    subprocess.call(["python", "streamline_data.py", file_path])
    file_path = "".join([data_path, '/large/', dataset, '.out'])
    subprocess.call(["python", "split_test_train.py", file_path])

  for pair, datasets in PAIR_TO_DATASET_NAMES.items():
    for dataset in datasets:
      print("".join([
        "python streamline_data.py ", data_path,'/small/', pair,
        "/", dataset, ".owl ", dataset]))

      file_path = "".join([data_path, '/small/', pair, '/', dataset, '.owl'])
      subprocess.call(["python", "streamline_data.py", file_path, dataset ])
      file_path = "".join([data_path, '/small/', pair, '/', dataset, '.out'])
      subprocess.call(["python", "split_test_train.py", file_path])

if __name__ == "__main__":
  main()
