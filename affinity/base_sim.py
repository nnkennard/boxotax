import nltk
import re

CAMEL_CASE_SPLITTER = re.compile('([A-Z][a-z]+)')

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



class Concept(object):
  def __init__(self, label, definition):
    self.label = label
    self.definition = definition
    self.normalize()
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

def main():
  print(treat_string("OnlyGoBackwards"))
  print(treat_string("onlyGoBackwards"))
  print(treat_string("only-go-backwards"))
  print(treat_string("only_go_backwards"))
  pass

if __name__ == "__main__":
  main()
