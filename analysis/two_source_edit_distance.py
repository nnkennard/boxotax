import collections
import nltk
from nltk import edit_distance
import sys

def main():
  mapping_file, mrconso_file = sys.argv[1:3]

  strings_to_find = collections.defaultdict(set)

  with open(mapping_file, 'r') as f:
    for line in f:
      source_id, target_id, _, umls_cid = line.strip().split()
      source_onto = source_id.split(":")[0]
      target_onto = target_id.split(":")[0]
      cid = umls_cid.split(":")[1]

      strings_to_find[cid].update([source_onto, target_onto])

  string_pairs = collections.defaultdict(
      lambda:collections.defaultdict(list))

  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      cid, source, string = fields[0],fields[11], fields[14].lower()
      if cid in strings_to_find.keys():
        if source in strings_to_find[cid]:
          string_pairs[cid][source].append(string)

  for cid, ontos in string_pairs.items():
    onto_keys = sorted(ontos.keys())
    for i, onto_1 in enumerate(onto_keys):
      for onto_2 in onto_keys[i+1:]:
        min_edit_distance = 10000
        min_edit_distance_pair = None
        for str_1 in ontos[onto_1]:
          for str_2 in ontos[onto_2]:
            new_edit_distance = edit_distance(str_1, str_2)
            if new_edit_distance < min_edit_distance:
              min_edit_distance = new_edit_distance
              min_edit_distance_pair = (str_1, str_2)
        print("\t".join([cid, onto_1, onto_2, str(min_edit_distance)] +
          list(min_edit_distance_pair)))


if __name__ == "__main__":
  main()
