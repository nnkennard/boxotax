import box_lib
import collections
import sys

class SourceMap(object):
  def __init__(self, source):
    self.source = source
    self.cui_map = collections.defaultdict(set)
    self.label_map = collections.defaultdict(set)
    self.finalized = False

  def finalize_label_map(self):
    new_label_map = {}
    for scui, labels in self.label_map.items():
      new_label_map[scui] = sorted(list(labels))

    del(self.label_map)

    self.label_map = new_label_map
    self.finalized = True

  def get_scuis(self, cui):
    return self.cui_map[cui]

  def get_scui_pairs(self, cui1, cui2):
    scui_pair_list = [(scui1, scui2) for scui1 in self.cui_map[cui1]
        for scui2 in self.cui_map[cui2]]
    return scui_pair_list

  def get_text_pairs(self, pairs):
    print(self.source)
    #assert self.finalized
    text_pairs = []
    for cui1, cui2 in pairs:
      for scui1, scui2 in self.get_scui_pairs(cui1, cui2):
        text_pairs.append((self.label_map[scui1][0], self.label_map[scui2][0]))
    return text_pairs


OBSOLETE_TTYS = ("MTH_OAF|MTH_OAP|MTH_OAS|MTH_OET|MTH_OF|MTH_OL|MTH_OPN"
                 "|MTH_OP|MTH_OS|IS|LO|MTH_IS|MTH_LO|OAF|OAM|OAP|OAS"
                 "|OA").split("|")

def get_pairs(source_map, pairs):
  text_pairs = []
  for hypo, hyper in pairs:
    text_pairs += source_map.get_text_pairs(hypo, hyper)
  return text_pairs


def main():
  umls_meta_path, output_dir = sys.argv[1:3]
  mrconso_file = umls_meta_path + "/MRCONSO.RRF"
  mrrel_file = umls_meta_path + "/MRREL.RRF"

  sources = {}
  for source in box_lib.UMLS_SOURCE_NAMES:
    sources[source] = SourceMap(source)

  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (cui, scui, sab, tty, label_str) = (
          fields[0], fields[10],  fields[11], fields[12], fields[14])
      if sab not in box_lib.UMLS_SOURCE_NAMES:
        continue
      if tty not in OBSOLETE_TTYS:
        if scui == "":
          scui = sab + ":" + cui
        sources[sab].cui_map[cui].add(scui)
        sources[sab].label_map[scui].add(label_str)

  for source_name in box_lib.UMLS_SOURCE_NAMES:
    print(source_name)
    sources[source].finalize_label_map()
    print(sources[source].finalized)

  pair_map = {source_name:list() for source_name in box_lib.UMLS_SOURCE_NAMES}

  with open(mrrel_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (hypo, rel, hyper, hypo_source, hyper_source) = (fields[0], fields[3],
          fields[4], fields[10], fields[11])
      if (hypo_source in box_lib.UMLS_SOURCE_NAMES
          and hypo_source == hyper_source
          and rel == "CHD"):
        pair_map[hypo_source].append((hypo, hyper))

  final_pairs = collections.defaultdict(list)

  for source_name in box_lib.UMLS_SOURCE_NAMES:
    final_pairs[source_name] += sources[source_name].get_text_pairs(
        pair_map[source_name])
  print(final_pairs)


if __name__ == "__main__":
  main()
