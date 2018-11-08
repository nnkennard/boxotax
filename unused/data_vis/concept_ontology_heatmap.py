import collections
import sys

def main():
  concept_map = collections.defaultdict(list)
  heat_map_builder = collections.defaultdict(lambda:
      collections.defaultdict(int))
  concepts_file = sys.argv[1]
  with open(concepts_file, 'r') as f:
    for line in f:
      concept, source = line.strip().split(';')
      concept_map[concept].append(source)

  for concept, sources in concept_map.iteritems():
    for source_1 in sources:
      for source_2 in sources:
        heat_map_builder[source_1][source_2] += 1

  for first_source, second_sources in heat_map_builder.iteritems():
    for second_source, count in second_sources.iteritems():
      print "\t".join([first_source, second_source, str(count)])

if __name__ =="__main__":
  main()
