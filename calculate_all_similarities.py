import box_lib
import subprocess
import sys


def main():
  train_data_path = sys.argv[1]
  for i, source1 in enumerate(box_lib.UMLS_SOURCE_NAMES):
    for source2 in box_lib.UMLS_SOURCE_NAMES[i+1:]:
      subprocess.call(["python", "calculate_similarities.py", train_data_path,
        source1, source2])


if __name__ == "__main__":
  main()
