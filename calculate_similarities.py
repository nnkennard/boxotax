import sys
import time
import tqdm

import base_sim

def get_paths(train_dir, source1, source2):
  file1 = train_dir + "/" + source1 + ".labels"
  file2 = train_dir + "/" + source2 + ".labels"
  output_file = train_dir + "/" + "-".join(sorted([source1, source2])) + ".sim"
  return file1, file2, output_file

def get_content_from_file(file_name):
  label_list = {}
  content_list = {}
  with open(file_name, 'r') as f:
    for line in f:
      fields = line.strip().split("\t")
      scui = fields[0]
      label_list[scui] = fields[1:]
      words = set(" ".join(fields[1:]).split())
      content = set(base_sim.get_content(words))
      content_list[scui] = content
  return label_list, content_list

def main():
  train_dir, source1, source2 = sys.argv[1:]
  file_1, file_2, output_file = get_paths(train_dir, source1, source2)

  labels1, content1 = get_content_from_file(file_1)
  labels2, content2 = get_content_from_file(file_2)

  with open(output_file, 'w') as f:
    pbar = tqdm.tqdm(total=len(content1))
    for scui1, contents1 in content1.items():
      for scui2, contents2 in content2.items():
        if contents1.intersection(contents2):
          sim = base_sim.simple_list_sim(labels1[scui1], labels2[scui2])
          f.write("\t".join([scui1, scui2, str(sim)]) + "\n")
      pbar.update(1)
    pbar.close()



if __name__ == "__main__":
  main()
