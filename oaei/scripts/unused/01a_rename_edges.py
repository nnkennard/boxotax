import pickle
import sys
import rdflib
import oaei_lib
import collections

# TODO: WRITE A TEST FOR THIS OMG

class Datasets(object):
  FMA = "fma"
  NCI = "nci"
  SNOMED = "snomed"

ALL_DATASETS = [Datasets.FMA, Datasets.NCI, Datasets.SNOMED]

PAIR_TO_DATASET_NAMES = {
    "fma2nci": [Datasets.FMA, Datasets.NCI],
    "fma2snomed": [Datasets.FMA, Datasets.SNOMED],
    "snomed2nci": [Datasets.NCI, Datasets.SNOMED]
    }

DATASET_PAIRS = PAIR_TO_DATASET_NAMES.keys()

ROOT_STR = "!!ROOT"
# I just want it to get index 0

def get_large_path(data_path, dataset):
  return "".join([data_path, '/large/', dataset, '.owl'])

def get_small_path(data_path, pair_name, dataset):
  return "".join([data_path, '/small/', pair_name, '/', dataset, '.owl'])

def print_pair(hyper, hypo, outfile):
  outfile.write("".join([hyper, '\t', hypo, '\n']))

def get_pairs_as_graph(dataset_prefix, file_thingy):
  g = rdflib.Graph()
  result = g.parse(file_thingy)
  graph = collections.defaultdict(set)

  for subj, pred, obj in g:
    if str(pred) == oaei_lib.RDFPredicates.SUBCLASS_OF:
      if oaei_lib.is_valid_label(subj) and oaei_lib.is_valid_label(obj):
        graph[oaei_lib.strip_prefix(obj)].add(oaei_lib.strip_prefix(subj))

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(set.union(*graph.values()))
  all_nodes = superclass_nodes.union(subclass_nodes)
  non_subclass_nodes = all_nodes - subclass_nodes

  graph[ROOT_STR] = list(non_subclass_nodes)
  all_nodes.add(ROOT_STR)
  for hyper, hypos in graph.items():
    for hypo in hypos:
      print_pair(hyper, hypo, sys.stdout)

  return graph



def main():
  data_path = sys.argv[1]

  large_nodes = {}
  small_nodes = collections.defaultdict(set)

  for dataset in ALL_DATASETS:
    with open(get_large_path(data_path, dataset), 'r') as f:
      _, nodes = get_pairs_as_graph(dataset, f)
      large_nodes[dataset] = nodes

  for pair_name, datasets in PAIR_TO_DATASET_NAMES.items():
    for dataset in datasets:
      with open(get_small_path(data_path, pair_name, dataset), 'r') as f:
        _, nodes = get_pairs_as_graph(f)
        small_nodes[dataset].update(nodes)

  for k, v in large_nodes.items():
    print(k,len(v))

  for k, v in small_nodes.items():
    print(k,len(v))


if __name__ == "__main__":
  main()
