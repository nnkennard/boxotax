import nltk
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
import sys

CAMEL_CASE_SPLITTER = re.compile('([A-Z][a-z]+)')
ENGLISH_STOPWORDS = stopwords.words('english')

def normalize(input_string):
  if "!!ROOT" in input_string:
    return [input_string]
  underscore_splitted_words = input_string.split("_")
  assert underscore_splitted_words[0] in ["fma", "nci", "snomed"]
  return underscore_splitted_words[1:]

def read_labels_from_file(filename):
  label_list = []
  with open(filename, 'r') as f:
    for line in f:
      label_list.append(line.strip())
  return label_list

def phrase_edit_distance(main_phrase, aux_phrase):
  # TODO : Rename because this is not edit distance!
  matches = 0
  total = 0
  for main_word in normalize(main_phrase):
    for aux_word in normalize(aux_phrase):
      if edit_distance(main_word, aux_word) < 2:
        matches += 1
      total += 1
  if matches == 0:
    return None
  else:
    return float(matches)/float(total)

def main():
  vocab_file = sys.argv[1]
  output_file = vocab_file.replace("vocab", "similarities")

  labels = read_labels_from_file(vocab_file)
  print(labels)

  weights = {}

  with open(output_file, 'w') as f:
    for main_label in labels:
      print(main_label)
      for aux_label in labels:
        print("  "+aux_label)
        if main_label.split('_')[0] == aux_label.split('_')[0]:
          continue
        key = "\t".join(
            sorted([str(labels.index(main_label)),
                    str(labels.index(aux_label))]))
        if key not in weights:
          edit_distance = phrase_edit_distance(main_label, aux_label)
          if edit_distance is not None:
            weight = str(edit_distance)
            f.write("\t".join([key, weight]) + "\n")
          else:
            weight = "0.0"
          weights[key] = weight


if __name__ == "__main__":
  main()
