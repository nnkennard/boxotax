import sys
import time
import tqdm

import base_sim

def get_paths(train_dir, source1, source2):
  file1 = train_dir + "/" + source1 + ".labels"
  file2 = train_dir + "/" + source2 + ".labels"
  output_file = train_dir + "/" + "-".join(sorted([source1, source2])) + ".sim"
  return file1, file2, output_file

def get_labels_from_file(file_name):
  label_list = []
  with open(file_name, 'r') as f:
    for line in f:
      label_list.append(line.strip().split("\t"))
  return label_list

def main():
  train_dir, source1, source2 = sys.argv[1:]
  file_1, file_2, output_file = get_paths(train_dir, source1, source2)

  labels1 = get_labels_from_file(file_1)
  labels2 = get_labels_from_file(file_2)

  with open(output_file, 'w') as f:
    pbar = tqdm.tqdm(total=len(labels1))
    for label_list1 in labels1:
      for label_list2 in labels2:
        sim = base_sim.simple_list_sim(label_list1, label_list2)
        f.write("\t".join([label_list1[0], label_list2[0], str(sim)]) + "\n")
      pbar.update(1)
    pbar.close()



if __name__ == "__main__":
  main()
