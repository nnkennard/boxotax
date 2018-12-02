import nltk
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
import sys

CAMEL_CASE_SPLITTER = re.compile('([A-Z][a-z]+)')
ENGLISH_STOPWORDS = stopwords.words('english')

def normalize(input_string):
  camel_case_split = re.split(CAMEL_CASE_SPLITTER, input_string)
  if '_' not in input_string and '-' not in input_string and len(camel_case_split) > 1:
    atoms = [word for word in camel_case_split if word]
  else:
    snake_case_split = input_string.split("_")
    if len(snake_case_split) > 1:
      atoms = snake_case_split
    else:
      fake_snake_case_split = input_string.split("-")
      if len(fake_snake_case_split) > 1:
        atoms = fake_snake_case_split
      else:
        atoms = [input_string]

  return list(set([atom.lower()
    for atom in atoms if atom not in ENGLISH_STOPWORDS]))

def read_labels_from_file(filename):
  label_list = []
  with open(filename, 'r') as f:
    for line in f:
      label_list.append(line.strip())
  return label_list

def phrase_edit_distance(main_phrase, aux_phrase):
  distances = []
  for main_word in normalize(main_phrase):
    for aux_word in normalize(aux_phrase):
      distances.append(edit_distance(main_word, aux_word))
      print(main_word, aux_word, edit_distance(main_word, aux_word))
  if len(distances) < 3:
    return float(sum(distances))/len(distances)
  else:
    return sum(sorted(distances)[-3:])/3.0

def main():
  main_label_file, aux_label_file = sys.argv[1:3]

  main_labels = read_labels_from_file(main_label_file)
  aux_labels = read_labels_from_file(aux_label_file)

  for main_label in main_labels:
    for aux_label in aux_labels:
      print("\t".join([main_label, aux_label,
        str(phrase_edit_distance(main_label, aux_label))]))

if __name__ == "__main__":
  main()
