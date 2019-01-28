
def lexical_matcher(source, target):
  alignment = []
  for name in source.names:
    if name in target:
      source_classes = source.get_classes_by_name(name)
      target_classes = target.get_classes_by_name(name)
      for source_class in source_classes:
        source_weight = source.get_weight(name, source_class)
        for target_class in target_classes:
          sim = source_weight * target.get_weight(name, target_class)
          alignment.append((source_class, target_class, sim))


# Needs external ontology
def mediating_matcher():
  pass

# Needs corpus?
def word_matcher():
  pass


# various string matchers from AML?
def parametric_string_matcher():
  pass


def definition_matcher():

