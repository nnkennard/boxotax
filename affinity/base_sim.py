import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re

CAMEL_CASE_SPLITTER = re.compile('([A-Z][a-z]+)')
ENGLISH_STOP_WORDS = stopwords.words('english')

def treat_string(input_string):
  camel_case_split = re.split(CAMEL_CASE_SPLITTER, input_string)
  if len(camel_case_split) > 1:
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

  return " ".join(atoms)

def remove_stopwords(input_string):
  return " ".join([word for word in input_string.split()
    if word not in ENGLISH_STOP_WORDS])

def get_definitions(label):
  definitions = []
  for word in label.split():
    print("WWWWW "+word)
    synsets = wordnet.synsets(word)
    if synsets:
      definitions += synsets[0].definition().split()
  return definitions


class Concept(object):
  def __init__(self, label, definition):
    self.label = label
    self.definition = definition
    #self.normalize()
    pass

  def normalize(self):
    self.label_n = normalize(self.label)
    self.definition_n = normalize(self.definition)

def base_sim(main_concept, aux_concept):
  main_label = main_concept.label
  aux_label = aux_concept.label

  # If labels are identical, return similarity of 1
  if main_label == aux_label:
    return 1
  main_label = treat_string(main_label)
  aux_label = treat_string(aux_label)
  if main_label == aux_label:
    return 1
  main_label = remove_stopwords(main_label)
  aux_label = remove_stopwords(aux_label)

  main_definitions = get_definitions(main_label)
  print(main_definitions)
  aux_definitions = get_definitions(aux_label)
  print(aux_definitions)

def main():
  print(treat_string("OnlyGoBackwards"))
  print(treat_string("onlyGoBackwards"))
  print(treat_string("only-go-backwards"))
  print(treat_string("only_go_backwards"))

  main_concept = Concept("CleanCanteen", "o")
  aux_concept = Concept("SplendidPlant", "p")

  print base_sim(main_concept, aux_concept)
  pass

if __name__ == "__main__":
  main()
