import box_lib
import sys

def add(vocab, fields, prefix):
  (cui, term_status, string_type, sab, tty, code, label_str) = (
      fields[0], fields[2], fields[4], fields[11], fields[12], fields[13], fields[14])
  if prefix + code not in vocab:
    vocab[prefix + code] = (cui, label_str)

def main():
  mrconso_file, source, output_dir = sys.argv[1:4]
  source_ab = box_lib.SAB_MAP[source]
  prefix = box_lib.PREFIX_MAP[source]
  vocab = {}
  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (cui, term_status, string_type, sab, tty, code, label_str) = (
          fields[0], fields[2], fields[4], fields[11], fields[12], fields[13], fields[14])
      if not sab == source_ab:
        continue
      if source == "msh":
        if (term_status == "P" and string_type == "VCW" and tty in ["NM"]):
            add(fields)
        if (term_status == "S" and string_type == "PF" and tty in ["MH", "NM",
        "ET", "TQ", "CE"]):
          add(fields)
        if (term_status == "P" and string_type == "VO" and tty in ["MH", "NM"]):
          add(fields)
        if (term_status == "P" and string_type == "VC" and tty in ["NM", "PM"]):
          add(fields)
        if (term_status == "P" and string_type == "PF" and tty in ["NM", "PEP",
          "MH", "ET", "TQ", "PM"]):
          add(fields)
        if (term_status == "P" and string_type == "VC" and tty in ["MH", "ET"]):
          add(fields)
          vocab[prefix + code] = label_str
      elif source == "go":
        if term_status == 'P':
          add(vocab, fields, prefix)
        if term_status == 'S':
          if tty in ["ET", "OP"]:
            add(vocab, fields, prefix)
        elif tty in ["PT", "IS"]:
          add(vocab, fields, prefix)
      elif source == "hpo":
        if term_status == 'S':
          if tty in ["ET", "OP"]:
            add(vocab, fields, prefix)
        elif tty in ["PT", "IS"]:
          add(vocab, fields, prefix)


  output_file = output_dir + "/" + source + ".names"
  with open(output_file, 'w') as f:
    for spec_key, (cui, label_str) in vocab.items():
      f.write("\t".join([spec_key, cui, label_str]) + "\n")


if __name__ == "__main__":
  main()
