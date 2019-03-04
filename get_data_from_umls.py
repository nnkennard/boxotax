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

    self.label_map = new_label_map
    self.finalized = True

  def get_scuis(self, cui):
    return self.cui_map[cui]

  def convert_scui_pairs(self, cui1, cui2):
    scui_pair_list = [(scui1, scui2) for scui1 in self.cui_map[cui1]
        for scui2 in self.cui_map[cui2]]
    return scui_pair_list

  def get_scui_pairs(self, pairs):
    text_pairs = []
    for cui1, cui2 in pairs:
      text_pairs += self.convert_scui_pairs(cui1, cui2)
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

  pair_map = {source_name:list() for source_name in box_lib.UMLS_SOURCE_NAMES}

  with open(mrrel_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (hyper, rel, hypo, hyper_source, hypo_source) = (fields[0], fields[3],
          fields[4], fields[10], fields[11])
      if (hypo_source in box_lib.UMLS_SOURCE_NAMES
          and hypo_source == hyper_source
          and rel == "CHD"):
        pair_map[hypo_source].append((hypo, hyper))

  for source_name in box_lib.UMLS_SOURCE_NAMES:
    source_map = sources[source_name]
    source_map.finalize_label_map()
    final_pairs = source_map.get_scui_pairs(pair_map[source_name])
    output_file = output_dir + "/" + source_name + ".txt"
    with open(output_file, 'w') as f:
      for hypo, hyper in final_pairs:
        f.write(hypo + "\t" + hyper + "\n")
    output_label_file = output_dir + "/" + source_name + ".labels"
    with open(output_label_file, 'w') as f:
      for scui, labels in source_map.label_map.items():
        f.write("\t".join([scui] + labels) + "\n")


if __name__ == "__main__":
  main()
