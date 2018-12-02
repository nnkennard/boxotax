import oaei_lib
import subprocess
import sys

def main():
  data_path = sys.argv[1]

  for pair, datasets in oaei_lib.PAIR_TO_DATASET_NAMES.items():
    for dataset in datasets:
      print("Streamlining " + pair + " " + dataset )
      file_path = "".join([data_path, '/small/', pair, '/', dataset, '.owl'])
      subprocess.call(
          ["python", "01a_streamline_data.py", file_path, dataset ])

      print("Splitting " + pair + " " + dataset)
      file_path = "".join([data_path, '/small/', pair, '/', dataset, '.out'])
      subprocess.call(
          ["python", "01b_split_test_train.py", file_path])

if __name__ == "__main__":
  main()
