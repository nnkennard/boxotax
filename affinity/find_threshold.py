import collections
import sys

import base_sim

def find_threshold():
  pass



def main():
  train_file_path, test_pair = sys.argv[1:]

  similarities = collections.defaultdict(set)

  for file_name in glob(train_file_path):
    if test_pair in file_name:
      continue
    else:
      with open(file_name, 'r') as f:
        for line in f:
          source_id, target_id, label = line.strip().split("\t")[:3]
          similarities[label].append(base_sim(source_id, target_id)


  
  pass


if __name__ == "__main__":
  main()
