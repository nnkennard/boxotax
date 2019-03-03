import box_lib
import collections
import sys

class SourceMap(object):
  def __init__(self, source):
    self.source = source
    self.cui_map = collections.defaultdict(list)
    self.label_map = collections.defaultdict(list)

OBSOLETE_TTYS = ("MTH_OAF|MTH_OAP|MTH_OAS|MTH_OET|MTH_OF|MTH_OL|MTH_OPN"
                 "|MTH_OP|MTH_OS|IS|LO|MTH_IS|MTH_LO|OAF|OAM|OAP|OAS"
                 "|OA").split("|")

def main():
  umls_meta_path, output_dir = sys.argv[1:3]
  mrconso_file = umls_meta_path + "/MRCONSO.RRF"
  mrref_file = umls_meta_path + "/MRREF.RRF"

  sources = {}
  for source in box_lib.UMLS_SOURCE_NAMES:
    sources[source] = SourceMap(source)

  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (cui, scui, sab, tty, label_str) = (
          fields[0], fields[10],  fields[11], fields[12], fields[14])
      if sab not in SOURCE_NAMES:
        continue
      if tty not in OBSOLETE_TTYS:
        if scui == "":
          scui = sab + ":" + cui
        sources[sab].cui_map[cui].append(scui)
        sources[sab].label_map[scui].append(label_str)

  pair_map = {source_name:list() for source_name in box_lib.SOURCE_NAMES}
  with open(mrrel_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (hypo, rel, hyper, hypo_source, hyper_source) = (fields[0], fields[3],
          fields[4], fields[10], fields[11])
      if (hypo_source in box_lib.UMLS_SOURCE_NAMES
          and hypo_source == hyper_source
          and rel == "CHD"):
        pair_map[hypo_source].append((hypo, hyper))

  print(pair_map["GO"])


if __name__ == "__main__":
  main()
